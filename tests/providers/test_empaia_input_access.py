"""EMPAIA inputs reach a builder through the provider's access adapter."""

from __future__ import annotations

import asyncio
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import httpx
from fastapi import FastAPI

from model_service.builders.application import BuilderApplication
from model_service.builders.client import HTTPBuilderDispatch, RemoteFailureError
from model_service.builders.contracts import (
    INVOCATION,
    BuilderFailure,
    BuilderResult,
    ExecutionContext,
    FailureCode,
    InlineOutput,
    InvalidInvocation,
    ProviderInput,
)
from model_service.builders.http import RunRequest, run_http
from model_service.builders.spec import InputSpec
from model_service.providers.empaia.gateway import EmpaiaGatewayError
from model_service.providers.empaia.input_access import EmpaiaInputAccessAdapter
from tests.support.builder_fixtures import (
    DECLARED,
    SPATIAL_MODEL,
    FunctionBuilder,
    inline_output,
    invocation,
    spec,
)


if TYPE_CHECKING:
    from model_service.builders.base import BuilderContext


class EmpaiaInputAccessTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancelled_empaia_acquisition_closes_gateway(self) -> None:
        started = asyncio.Event()
        closed = asyncio.Event()

        class Gateway:
            async def materialize_input(
                self,
                _name: str,
                _directory: str,
                *,
                mpp: float | None = None,
                **_: object,
            ) -> str:
                started.set()
                await asyncio.Event().wait()

            async def close(self) -> None:
                closed.set()
                raise OSError("cleanup failure")

        with TemporaryDirectory() as directory:
            adapter = EmpaiaInputAccessAdapter(lambda _c: Gateway(), Path(directory))
            request = invocation(
                "empaia",
                {"api_url": "https://example.invalid", "job_id": "job"},
                inputs={"slide": ProviderInput("slide")},
            )
            task = asyncio.create_task(
                adapter.create(
                    request,
                    ExecutionContext(
                        processing_attempt_id="attempt",
                        protected_context={
                            "api_url": "https://example.invalid",
                            "job_id": "job",
                            "credential": "secret",
                        },
                    ),
                )
            )
            await asyncio.wait_for(started.wait(), timeout=1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1)
            self.assertTrue(closed.is_set())

    async def test_empaia_input_failures_keep_the_gateways_reason(self) -> None:
        cases = [
            EmpaiaGatewayError(
                "EMPAIA AppService is temporarily unavailable", retryable=True
            ),
            EmpaiaGatewayError(
                "EMPAIA slide region is not a readable image", retryable=False
            ),
        ]
        for failure in cases:

            class Gateway:
                async def materialize_input(
                    self,
                    _name: str,
                    _directory: str,
                    *,
                    mpp: float | None = None,
                    failure: EmpaiaGatewayError = failure,
                    **_: object,
                ) -> str:
                    raise failure

                async def close(self) -> None:
                    return None

            with (
                self.subTest(message=failure.safe_message),
                TemporaryDirectory() as directory,
            ):
                adapter = EmpaiaInputAccessAdapter(
                    lambda _c: Gateway(), Path(directory)
                )
                with self.assertRaises(BuilderFailure) as raised:
                    await adapter.create(
                        invocation(
                            "empaia",
                            {"api_url": "https://example.invalid", "job_id": "job"},
                            inputs={"slide": ProviderInput("slide")},
                        ),
                        ExecutionContext(
                            processing_attempt_id="attempt",
                            protected_context={
                                "api_url": "https://example.invalid",
                                "job_id": "job",
                                "credential": "secret",
                            },
                        ),
                    )
                self.assertEqual(raised.exception.code, FailureCode.INPUT_UNAVAILABLE)
                self.assertEqual(list(Path(directory).rglob("inputs")), [])
                self.assertEqual(raised.exception.safe_message, failure.safe_message)
                self.assertEqual(raised.exception.retryable, failure.retryable)

    async def test_empaia_access_validates_contexts_before_opening_a_gateway(
        self,
    ) -> None:
        def unexpected_gateway(_context: object) -> Any:
            self.fail("invalid context must not open an AppService gateway")

        public = {"api_url": "https://empaia.invalid", "job_id": "job"}
        protected = {**public, "credential": "secret"}
        runtime = ExecutionContext(
            processing_attempt_id="attempt", protected_context=protected
        )
        cases = [
            ({}, runtime),
            ({**public, "job_id": False}, runtime),
            ({**public, "credential": "secret"}, runtime),
            ({**public, "processing_attempt_id": "attempt"}, runtime),
            ({**public, "job_id": "another-job"}, runtime),
            ({**public, "api_url": "https://another.invalid"}, runtime),
            (public, ExecutionContext(processing_attempt_id="attempt")),
            (public, ExecutionContext(protected_context=protected)),
            (
                public,
                ExecutionContext(
                    processing_attempt_id="attempt",
                    protected_context={**protected, "credential": 1},
                ),
            ),
        ]
        with TemporaryDirectory() as directory:
            adapter = EmpaiaInputAccessAdapter(unexpected_gateway, Path(directory))
            for state, context in cases:
                with (
                    self.subTest(state=state),
                    self.assertRaises(InvalidInvocation) as raised,
                ):
                    await adapter.create(invocation("empaia", state), context)
                self.assertNotIn("secret", str(raised.exception))

    async def test_dispatch_keeps_runtime_context_out_of_the_durable_invocation(
        self,
    ) -> None:
        seen_credentials: list[str] = []
        closed: list[bool] = []
        seen_requests: list[RunRequest] = []

        class Gateway:
            async def materialize_input(
                self,
                _name: str,
                directory: str,
                *,
                mpp: float | None = None,
                **_: object,
            ) -> str:
                path = Path(directory) / "slide.svs"
                path.write_bytes(b"slide")
                return str(path)

            async def close(self) -> None:
                closed.append(True)

        def gateway_factory(context: Any) -> Gateway:
            seen_credentials.append(context.credential)
            return Gateway()

        async def build(context: BuilderContext) -> BuilderResult:
            context.inputs.slide  # type: ignore[attr-defined]  # noqa: B018
            return BuilderResult({"score": InlineOutput(1)})

        with TemporaryDirectory() as directory:
            application = BuilderApplication(
                spec(inputs={"slide": InputSpec()}, outputs={"score": inline_output()}),
                FunctionBuilder(build),
                {"empaia": EmpaiaInputAccessAdapter(gateway_factory, Path(directory))},
                model_handles=DECLARED,
            )
            api = FastAPI()

            @api.post("/builders/probe/", response_model=BuilderResult)
            async def endpoint(body: RunRequest) -> BuilderResult:
                seen_requests.append(body)
                return await run_http(application, body)

            public = {"api_url": "https://empaia.invalid", "job_id": "job"}
            request = invocation(
                "empaia", public, inputs={"slide": ProviderInput("slide")}
            )
            before = INVOCATION.dump_python(request, mode="json")
            runtime = ExecutionContext(
                processing_attempt_id="attempt",
                protected_context={**public, "credential": "secret"},
            )
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=api)
            ) as client:
                dispatch = HTTPBuilderDispatch(client, origin="http://ray")
                result = await dispatch.run("probe", request, SPATIAL_MODEL, runtime)
                with self.assertRaises(RemoteFailureError) as raised:
                    await dispatch.run(
                        "probe",
                        request,
                        SPATIAL_MODEL,
                        ExecutionContext(processing_attempt_id="attempt"),
                    )

        self.assertEqual(result, BuilderResult({"score": InlineOutput(1)}))
        self.assertEqual(seen_requests[0].execution_context, runtime)
        self.assertEqual(seen_requests[0].model, SPATIAL_MODEL)
        self.assertEqual(
            INVOCATION.dump_python(seen_requests[0].invocation, mode="json"), before
        )
        self.assertEqual(seen_credentials, ["secret"])
        self.assertEqual(closed, [True])
        self.assertEqual(raised.exception.code, "invalid_builder_invocation")
        self.assertFalse(raised.exception.retryable)
        self.assertNotIn("secret", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
