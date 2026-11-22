"""Check builder dispatch over HTTP: request shape and outcome classification."""

from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import httpx

from model_service.api.scheduler.builder_dispatch import (
    ConfirmedRejectionError,
    HTTPBuilderDispatch,
    RemoteFailureError,
    UnknownOutcomeError,
)
from model_service.contracts.analysis_http import AnalysisRunRequest
from model_service.contracts.invocation import (
    ExecutionContext,
    InputAccessContext,
    Invocation,
    SharedPathInput,
)
from model_service.contracts.model_declarations import SpatialScoresModel
from model_service.contracts.result import BuilderResult, InlineOutput


if TYPE_CHECKING:
    from collections.abc import Callable


MODEL = SpatialScoresModel(
    id="prostate-classifier-1",
    tile_size=512,
    output_tile_size=1,
    n_channels=1,
    mpp=0.468,
)


def respond(response: httpx.Response) -> Callable[[httpx.Request], httpx.Response]:
    return lambda _request: response


class HttpBuilderDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def dispatch(
        self,
        handler: Callable[[httpx.Request], httpx.Response],
        context: ExecutionContext | None = None,
    ) -> BuilderResult:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await HTTPBuilderDispatch(client, origin="http://ray:8000/").run(
                "heatmap",
                Invocation(
                    inputs={"slide": SharedPathInput(path="slide.svs")},
                    configuration=None,
                    input_access=InputAccessContext(
                        kind="shared_storage", state={"root": "/mnt/data"}
                    ),
                    deadline_at=datetime.now(UTC) + timedelta(minutes=1),
                ),
                MODEL,
                context or ExecutionContext(),
            )

    async def test_success_sends_invocation_model_and_context(self) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200, json={"outputs": {"score": {"kind": "inline", "value": 0.9}}}
            )

        result = await self.dispatch(
            handler,
            ExecutionContext(
                processing_attempt_id="attempt-1",
                execution_id="execution-1",
                protected_context={"credential": "secret"},
            ),
        )

        self.assertEqual(result, BuilderResult(outputs={"score": InlineOutput(0.9)}))
        [request] = requests
        self.assertEqual(request.url, "http://ray:8000/builders/heatmap/")
        body = json.loads(request.content)
        self.assertEqual(body["invocation"]["inputs"]["slide"]["path"], "slide.svs")
        self.assertIsNone(body["invocation"]["configuration"])
        self.assertEqual(body["model"]["id"], "prostate-classifier-1")
        self.assertEqual(body["model"]["contract"], "spatial-scores")
        self.assertEqual(
            body["execution_context"],
            {
                "processing_attempt_id": "attempt-1",
                "execution_id": "execution-1",
                "protected_context": {"credential": "secret"},
            },
        )
        self.assertLessEqual(request.extensions["timeout"]["read"], 60)

    async def test_ray_serves_own_503_is_a_confirmed_rejection(self) -> None:
        response = httpx.Response(503, text="queue full", headers={"retry-after": "7"})
        with self.assertRaises(ConfirmedRejectionError) as raised:
            await self.dispatch(respond(response))

        self.assertEqual(raised.exception.retry_after, timedelta(seconds=7))

    async def test_backpressure_is_a_confirmed_rejection(self) -> None:
        for headers, retry_after in (
            ({"retry-after": "7"}, timedelta(seconds=7)),
            ({"retry-after": "soon"}, None),
            ({}, None),
        ):
            response = httpx.Response(429, headers=headers)
            with self.subTest(headers=headers):
                with self.assertRaises(ConfirmedRejectionError) as raised:
                    await self.dispatch(respond(response))
                self.assertEqual(raised.exception.retry_after, retry_after)

    async def test_classified_failures_keep_the_safe_error(self) -> None:
        detail = {
            "code": "model_contract_mismatch",
            "message": "model changed",
            "retryable": False,
            "future_metadata": "ignored",
        }

        with self.assertRaises(RemoteFailureError) as raised:
            await self.dispatch(
                lambda _r: httpx.Response(422, json={"detail": detail, "extra": 1})
            )

        self.assertEqual(raised.exception.code, "model_contract_mismatch")
        self.assertEqual(raised.exception.safe_message, "model changed")
        self.assertFalse(raised.exception.retryable)

    async def test_transport_errors_and_malformed_responses_are_unknown(self) -> None:
        def transport_error(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("lost", request=request)

        valid = {"code": "failed", "message": "failed", "retryable": True}
        bodies: list[object] = [[], "failure", {}, {"detail": None}]
        for name, values in (
            ("code", (None, "", 1)),
            ("message", (None, "", 1)),
            ("retryable", (None, "true", 0)),
        ):
            bodies.extend({"detail": {**valid, name: value}} for value in values)
        handlers: list[Callable[[httpx.Request], httpx.Response]] = [
            transport_error,
            lambda _r: httpx.Response(200, json={"outputs": {"x": {"kind": "?"}}}),
            *(respond(httpx.Response(503, json=body)) for body in bodies),
        ]
        for index, handler in enumerate(handlers):
            with self.subTest(index), self.assertRaises(UnknownOutcomeError):
                await self.dispatch(handler)


if __name__ == "__main__":
    unittest.main()


class HttpAnalysisDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def dispatch(
        self, handler: Callable[[httpx.Request], httpx.Response]
    ) -> BuilderResult:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await HTTPBuilderDispatch(
                client, origin="http://ray:8000/"
            ).run_analysis(
                AnalysisRunRequest(
                    invocation=Invocation(
                        inputs={"slide": SharedPathInput(path="slide.svs")},
                        configuration={"stride_fraction": 1.0},
                        input_access=InputAccessContext(
                            kind="shared_storage", state={"root": "/mnt/data"}
                        ),
                        deadline_at=datetime.now(UTC) + timedelta(minutes=1),
                    ),
                    analysis={"id": "prostate-heatmap", "digest": "abc"},
                )
            )

    async def test_a_run_goes_to_the_runner_with_the_resolved_document(
        self,
    ) -> None:
        requests: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            return httpx.Response(
                200, json={"outputs": {"score": {"kind": "inline", "value": 0.9}}}
            )

        result = await self.dispatch(handler)

        self.assertEqual(result, BuilderResult(outputs={"score": InlineOutput(0.9)}))
        [request] = requests
        self.assertEqual(request.url, "http://ray:8000/analyses/")
        body = json.loads(request.content)
        self.assertEqual(body["analysis"], {"id": "prostate-heatmap", "digest": "abc"})
        self.assertEqual(body["invocation"]["inputs"]["slide"]["path"], "slide.svs")
        self.assertLessEqual(request.extensions["timeout"]["read"], 60)

    async def test_the_runners_answers_are_classified_like_a_builders(self) -> None:
        mismatch = httpx.Response(
            422,
            json={
                "detail": {
                    "code": "model_contract_mismatch",
                    "message": "model changed",
                    "retryable": False,
                }
            },
        )
        with self.assertRaises(RemoteFailureError) as raised:
            await self.dispatch(respond(mismatch))
        with self.assertRaises(ConfirmedRejectionError):
            await self.dispatch(respond(httpx.Response(429)))

        self.assertEqual(raised.exception.code, "model_contract_mismatch")
