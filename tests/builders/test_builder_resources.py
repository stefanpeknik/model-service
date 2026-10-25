"""Check deadline-aware acquisition, bounded cleanup, cancellation and attempt isolation."""

from __future__ import annotations

import asyncio
import gc
import unittest
import weakref
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from model_service.builders.application import BuilderApplication
from model_service.builders.cleanup import close_resource
from model_service.builders.contracts import (
    BuilderFailure,
    BuilderResult,
    ExecutionContext,
    FailureCode,
    InputAccessContext,
    InvalidInvocation,
    ProviderInput,
    SharedArtifact,
)
from model_service.builders.registry import BUILDERS, get_builder
from model_service.providers.empaia.gateway import EmpaiaGatewayError
from model_service.providers.empaia.input_access import EmpaiaInputAccessAdapter
from model_service.providers.shared_storage import SharedStorageAccessAdapter
from model_service.providers.standalone import StandaloneProviderAdapter
from model_service.scheduler.active_work import ActiveWorkRegistry
from model_service.scheduler.contracts import ExecutionStatus, ScheduleRequest
from model_service.scheduler.scheduler import ExecutionScheduler
from model_service.scheduler.worker import ExecutionWorker
from tests.support.builder_fixtures import (
    SPATIAL_MODEL,
    FunctionBuilder,
    artifact_output,
    invocation,
    spec,
)
from tests.support.memory_repository import InMemoryExecutionRepository


if TYPE_CHECKING:
    from model_service.builders.application import BuilderContext
    from model_service.builders.contracts import Invocation, ModelBinding


async def empty(_context: BuilderContext) -> BuilderResult:
    return BuilderResult({})


def application(builder: Any, adapter: Any, **options: Any) -> BuilderApplication:
    return BuilderApplication(spec(), builder, {"test": adapter}, **options)


class Adapter:
    def __init__(self, access: Any) -> None:
        self._access = access

    def create(self, _invocation: object, _context: object, _model: object) -> Any:
        return self._access


class BuilderResourceTests(unittest.IsolatedAsyncioTestCase):
    async def test_deadline_includes_input_acquisition(self) -> None:
        cancelled = asyncio.Event()

        class BlockingAdapter:
            async def create(
                self, _invocation: object, _context: object, _model: object
            ) -> Any:
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

        request = invocation(
            "test", deadline_at=datetime.now(UTC) + timedelta(milliseconds=20)
        )
        with self.assertRaises(BuilderFailure) as raised:
            await asyncio.wait_for(
                application(FunctionBuilder(empty), BlockingAdapter()).run(
                    request, SPATIAL_MODEL
                ),
                timeout=1,
            )
        self.assertEqual(raised.exception.code, "builder_deadline_exceeded")
        self.assertFalse(raised.exception.retryable)
        self.assertTrue(cancelled.is_set())

    async def test_cleanup_failure_preserves_primary_failure(self) -> None:
        class Access:
            async def aclose(self) -> None:
                raise OSError("cleanup failure")

        for primary in (
            InvalidInvocation("invalid configuration"),
            BuilderFailure(code="permanent", safe_message="cannot", retryable=False),
        ):

            async def build(_context: BuilderContext, primary: Any = primary) -> Any:
                raise primary

            with (
                self.subTest(failure=type(primary).__name__),
                self.assertRaises(type(primary)) as raised,
            ):
                await application(FunctionBuilder(build), Adapter(Access())).run(
                    invocation("test"), SPATIAL_MODEL
                )
            self.assertIs(raised.exception, primary)

    async def test_repeated_cancellation_waits_for_cleanup(self) -> None:
        started = asyncio.Event()
        closing = asyncio.Event()
        release_close = asyncio.Event()
        closed = asyncio.Event()

        async def build(_context: BuilderContext) -> BuilderResult:
            started.set()
            await asyncio.Event().wait()
            raise AssertionError

        class Access:
            async def aclose(self) -> None:
                closing.set()
                await release_close.wait()
                closed.set()

        task = asyncio.create_task(
            application(FunctionBuilder(build), Adapter(Access())).run(
                invocation("test"), SPATIAL_MODEL
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        task.cancel()
        await asyncio.wait_for(closing.wait(), timeout=1)
        task.cancel()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        release_close.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
        self.assertTrue(closed.is_set())

    async def test_cleanup_is_bounded(self) -> None:
        close_cancelled = asyncio.Event()

        class Access:
            async def aclose(self) -> None:
                try:
                    await asyncio.Event().wait()
                finally:
                    close_cancelled.set()

        with self.assertRaises(BuilderFailure) as raised:
            await asyncio.wait_for(
                application(
                    FunctionBuilder(empty),
                    Adapter(Access()),
                    cleanup_timeout_seconds=0.02,
                ).run(invocation("test"), SPATIAL_MODEL),
                timeout=1,
            )
        self.assertEqual(raised.exception.code, "builder_internal_error")
        await asyncio.wait_for(close_cancelled.wait(), timeout=1)

    async def test_resource_self_cancellation_is_a_cleanup_failure(self) -> None:
        class Access:
            async def aclose(self) -> None:
                raise asyncio.CancelledError

        with self.assertRaises(BuilderFailure) as raised:
            await application(FunctionBuilder(empty), Adapter(Access())).run(
                invocation("test"), SPATIAL_MODEL
            )
        self.assertEqual(raised.exception.code, "builder_internal_error")

        async def build(_context: BuilderContext) -> BuilderResult:
            raise InvalidInvocation("primary failure")

        with self.assertRaisesRegex(InvalidInvocation, "primary failure"):
            await application(FunctionBuilder(build), Adapter(Access())).run(
                invocation("test"), SPATIAL_MODEL
            )

    async def test_timed_out_cleanup_remains_owned_until_it_stops(self) -> None:
        cancellation_received = asyncio.Event()
        closed = asyncio.Event()
        task_ref: weakref.ReferenceType[asyncio.Task[Any]] | None = None

        async def close() -> None:
            nonlocal task_ref
            current = asyncio.current_task()
            assert current is not None
            task_ref = weakref.ref(current)
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                cancellation_received.set()
                try:
                    await asyncio.Future()
                finally:
                    closed.set()

        with self.assertRaises(TimeoutError):
            await close_resource(close, timeout_seconds=0.02)
        await asyncio.wait_for(cancellation_received.wait(), timeout=1)
        gc.collect()
        assert task_ref is not None
        task = task_ref()
        self.assertIsNotNone(task)
        assert task is not None
        task.cancel()
        await asyncio.wait_for(closed.wait(), timeout=1)

    async def test_cancelled_empaia_acquisition_closes_gateway(self) -> None:
        started = asyncio.Event()
        closed = asyncio.Event()

        class Gateway:
            async def materialize_input(
                self, _name: str, _directory: str, *, mpp: float | None = None
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

    async def test_worker_supplies_the_attempt_and_isolates_results(self) -> None:
        with TemporaryDirectory() as directory:
            attempts: list[str | None] = []

            async def build(context: BuilderContext) -> BuilderResult:
                output = context.output_path("artifact")
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(str(len(attempts)).encode())
                return BuilderResult({"artifact": SharedArtifact(str(output))})

            builder_spec = spec(outputs={"artifact": artifact_output()})
            shared_storage = SharedStorageAccessAdapter([Path(directory)])
            builder = BuilderApplication(
                builder_spec, FunctionBuilder(build), {"shared_storage": shared_storage}
            )

            class Dispatch:
                async def run(
                    self,
                    builder_id: str,
                    request: Invocation,
                    model: ModelBinding,
                    execution_context: ExecutionContext,
                ) -> BuilderResult:
                    attempts.append(execution_context.processing_attempt_id)
                    assert request.input_access.state == {"root": directory}
                    return await builder.run(request, model, execution_context)

            def resolve(builder_id: str, model_id: str) -> ModelBinding:
                get_builder(builder_id)
                assert model_id == SPATIAL_MODEL.id
                return SPATIAL_MODEL

            repository = InMemoryExecutionRepository()
            provider = StandaloneProviderAdapter(shared_storage)
            with patch.dict(BUILDERS, {"test-builder": builder_spec}):
                scheduler = ExecutionScheduler(
                    repository=repository,
                    dispatch=Dispatch(),
                    providers={provider.name: provider},
                    resolve_model=resolve,
                    default_timeout=timedelta(minutes=1),
                    maximum_timeout=timedelta(minutes=1),
                )
                worker = ExecutionWorker(
                    repository=repository,
                    active_work=ActiveWorkRegistry(),
                    providers={provider.name: provider},
                    dispatch=Dispatch(),
                    resolve_model=resolve,
                    owner="test-worker",
                    lease_duration=timedelta(seconds=30),
                )
                scheduled = []
                for key in ("first", "second"):
                    scheduled.append(
                        await scheduler.schedule(
                            ScheduleRequest(
                                builder_id="test-builder",
                                model_id=SPATIAL_MODEL.id,
                                provider="standalone",
                                body={
                                    "inputs": {},
                                    "input_access": {
                                        "kind": "shared_storage",
                                        "state": {"root": directory},
                                    },
                                },
                                idempotency_key=key,
                            )
                        )
                    )
                for _pass in range(4):
                    self.assertTrue(await worker.run_once())
                results = [await scheduler.get(item.execution.id) for item in scheduled]

            self.assertTrue(
                all(item.status is ExecutionStatus.SUCCEEDED for item in results)
            )
            paths = []
            for item in results:
                assert item.result is not None
                artifact = item.result.outputs["artifact"]
                assert isinstance(artifact, SharedArtifact)
                paths.append(Path(artifact.path))
            self.assertNotEqual(paths[0], paths[1])
            self.assertTrue(all(".attempts" in path.parts for path in paths))
            self.assertEqual(paths[0].read_bytes(), b"1")
            self.assertEqual(paths[1].read_bytes(), b"2")
            self.assertEqual(len(set(attempts)), 2)
            self.assertNotIn(None, attempts)


class InputAccessContextTests(unittest.TestCase):
    def test_execution_context_rejects_invalid_attempts_and_non_json_secrets(
        self,
    ) -> None:
        from pydantic import TypeAdapter, ValidationError

        for attempt in ("", True, 1):
            with self.subTest(attempt=attempt), self.assertRaises(ValidationError):
                ExecutionContext(processing_attempt_id=attempt)  # type: ignore[arg-type]
        with self.assertRaises(ValidationError):
            ExecutionContext(protected_context={"invalid": float("nan")})
        with self.assertRaises(ValidationError):
            TypeAdapter(ExecutionContext).validate_python({"unexpected": True})

    def test_execution_context_representation_hides_credentials(self) -> None:
        context = ExecutionContext(protected_context={"credential": "secret-token"})
        self.assertNotIn("secret-token", repr(context))
        self.assertEqual(
            InputAccessContext("empaia", {}), InputAccessContext("empaia", {})
        )


if __name__ == "__main__":
    unittest.main()
