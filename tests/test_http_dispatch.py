"""Check builder dispatch over HTTP: request shape and outcome classification."""

from __future__ import annotations

import json
import unittest
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import httpx

from builders.client import (
    ConfirmedRejectionError,
    HTTPBuilderDispatch,
    RemoteFailureError,
    UnknownOutcomeError,
)
from builders.contracts import (
    BuilderResult,
    ExecutionContext,
    InlineOutput,
    InputAccessContext,
    Invocation,
    SharedPathInput,
    SpatialScoresModel,
)


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
                    resolved_configuration=None,
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
                protected_context={"credential": "secret"},
            ),
        )

        self.assertEqual(result, BuilderResult(outputs={"score": InlineOutput(0.9)}))
        [request] = requests
        self.assertEqual(request.url, "http://ray:8000/builders/heatmap/")
        body = json.loads(request.content)
        self.assertEqual(body["invocation"]["inputs"]["slide"]["path"], "slide.svs")
        self.assertIsNone(body["invocation"]["resolved_configuration"])
        self.assertEqual(body["model"]["id"], "prostate-classifier-1")
        self.assertEqual(body["model"]["contract"], "spatial-scores")
        self.assertEqual(
            body["execution_context"],
            {
                "processing_attempt_id": "attempt-1",
                "protected_context": {"credential": "secret"},
            },
        )
        self.assertLessEqual(request.extensions["timeout"]["read"], 60)

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
        bodies: list[object] = [None, [], "failure", {}, {"detail": None}]
        for name, values in (
            ("code", (None, "", 1)),
            ("message", (None, "", 1)),
            ("retryable", (None, "true", 0)),
        ):
            bodies.extend({"detail": {**valid, name: value}} for value in values)
        handlers: list[Callable[[httpx.Request], httpx.Response]] = [
            transport_error,
            lambda _r: httpx.Response(503, text="proxy unavailable"),
            lambda _r: httpx.Response(200, json={"outputs": {"x": {"kind": "?"}}}),
            *(respond(httpx.Response(503, json=body)) for body in bodies),
        ]
        for index, handler in enumerate(handlers):
            with self.subTest(index), self.assertRaises(UnknownOutcomeError):
                await self.dispatch(handler)


if __name__ == "__main__":
    unittest.main()
