from __future__ import annotations

import asyncio
import gc
import unittest
import weakref
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from execution_scheduler.contracts import ExecutionStatus, ScheduleRequest
from execution_scheduler.dispatch import InMemoryWorkflowDispatch
from execution_scheduler.repository import InMemoryExecutionRepository
from execution_scheduler.scheduler import ExecutionScheduler
from execution_scheduler.standalone import StandaloneProviderAdapter
from execution_scheduler.worker import ExecutionWorker
from integrations.empaia.input_access import EmpaiaInputAccessAdapter
from workflow_execution.application import WorkflowApplication
from workflow_execution.catalog import (
    WorkflowCatalog,
    WorkflowConfiguration,
    WorkflowDefinition,
    WorkflowValueDefinition,
    validate_empty_configuration,
)
from workflow_execution.contracts import (
    InputAccessContext,
    InvalidWorkflowInvocation,
    ProviderInput,
    SharedArtifact,
    WorkflowFailure,
    WorkflowInvocation,
    WorkflowResult,
)
from workflow_execution.resources import close_resource
from workflow_execution.serialization import invocation_to_dict
from workflow_execution.shared_storage import SharedStorageAccessAdapter


class WorkflowResourceTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def invocation() -> WorkflowInvocation:
        return WorkflowInvocation(
            inputs={},
            resolved_configuration={},
            input_access=InputAccessContext(kind="test", state={}),
            deadline_at=datetime.now(UTC) + timedelta(minutes=1),
        )

    async def test_deadline_includes_input_acquisition(self) -> None:
        cancelled = asyncio.Event()

        class Adapter:
            async def create(self, _invocation):
                try:
                    await asyncio.Event().wait()
                finally:
                    cancelled.set()

        application = WorkflowApplication(
            WorkflowDefinition("test", {}, {}),
            EmptyWorkflow(),
            access_adapters={"test": Adapter()},
        )
        invocation = replace(
            self.invocation(),
            deadline_at=datetime.now(UTC) + timedelta(milliseconds=20),
        )
        with self.assertRaises(WorkflowFailure) as raised:
            await asyncio.wait_for(application.run(invocation), timeout=1)
        self.assertEqual(raised.exception.code, "workflow_deadline_exceeded")
        self.assertFalse(raised.exception.retryable)
        self.assertTrue(cancelled.is_set())

    async def test_cleanup_preserves_primary_failure(self) -> None:
        class Workflow:
            async def run(self, _context):
                raise primary

        class Access:
            async def aclose(self):
                raise OSError("cleanup failure")

        class Adapter:
            def create(self, _invocation):
                return Access()

        application = WorkflowApplication(
            WorkflowDefinition("test", {}, {}),
            Workflow(),
            access_adapters={"test": Adapter()},
        )
        for primary in (
            InvalidWorkflowInvocation("invalid workflow configuration"),
            WorkflowFailure(
                code="permanent_workflow_failure",
                safe_message="workflow cannot complete",
                retryable=False,
            ),
        ):
            with self.subTest(failure=type(primary).__name__):
                with self.assertRaises(type(primary)) as raised:
                    await application.run(self.invocation())
                self.assertIs(raised.exception, primary)

    async def test_repeated_cancellation_waits_for_cleanup(self) -> None:
        started = asyncio.Event()
        closing = asyncio.Event()
        release_close = asyncio.Event()
        closed = asyncio.Event()

        class Workflow:
            async def run(self, _context):
                started.set()
                await asyncio.Event().wait()

        class Access:
            async def aclose(self):
                closing.set()
                await release_close.wait()
                closed.set()

        class Adapter:
            def create(self, _invocation):
                return Access()

        application = WorkflowApplication(
            WorkflowDefinition("test", {}, {}),
            Workflow(),
            access_adapters={"test": Adapter()},
        )
        task = asyncio.create_task(application.run(self.invocation()))
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
            async def aclose(self):
                try:
                    await asyncio.Event().wait()
                finally:
                    close_cancelled.set()

        class Adapter:
            def create(self, _invocation):
                return Access()

        application = WorkflowApplication(
            WorkflowDefinition("test", {}, {}),
            EmptyWorkflow(),
            access_adapters={"test": Adapter()},
            cleanup_timeout_seconds=0.02,
        )
        with self.assertRaises(WorkflowFailure) as raised:
            await asyncio.wait_for(application.run(self.invocation()), timeout=1)
        self.assertEqual(raised.exception.code, "workflow_internal_error")
        await asyncio.wait_for(close_cancelled.wait(), timeout=1)

    async def test_resource_self_cancellation_is_a_cleanup_failure(self) -> None:
        class Access:
            async def aclose(self):
                raise asyncio.CancelledError

        class Adapter:
            def create(self, _invocation):
                return Access()

        class Workflow:
            async def run(self, _context):
                raise InvalidWorkflowInvocation("primary failure")

        for workflow in (EmptyWorkflow(), Workflow()):
            application = WorkflowApplication(
                WorkflowDefinition("test", {}, {}),
                workflow,
                access_adapters={"test": Adapter()},
            )
            with self.subTest(workflow=type(workflow).__name__):
                if isinstance(workflow, EmptyWorkflow):
                    with self.assertRaises(WorkflowFailure) as raised:
                        await application.run(self.invocation())
                    self.assertEqual(raised.exception.code, "workflow_internal_error")
                else:
                    with self.assertRaisesRegex(
                        InvalidWorkflowInvocation, "primary failure"
                    ):
                        await application.run(self.invocation())

    async def test_timed_out_cleanup_remains_owned_until_it_stops(self) -> None:
        cancellation_received = asyncio.Event()
        closed = asyncio.Event()
        task_ref = None

        async def close():
            nonlocal task_ref
            task_ref = weakref.ref(asyncio.current_task())
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
        task = task_ref()
        self.assertIsNotNone(task)
        task.cancel()
        await asyncio.wait_for(closed.wait(), timeout=1)

    async def test_scheduled_artifacts_are_isolated_by_attempt(self) -> None:
        with TemporaryDirectory() as directory:
            adapter = SharedStorageAccessAdapter()

            def access(attempt):
                return adapter.create(
                    replace(
                        self.invocation(),
                        input_access=InputAccessContext(
                            kind="shared_storage",
                            state={
                                "root": directory,
                                "processing_attempt_id": attempt,
                            },
                        ),
                    )
                )

            old_path = access("old").output_path("outputs/heatmap.tiff")
            new_path = access("new").output_path("outputs/heatmap.tiff")
            self.assertNotEqual(old_path, new_path)
            new_path.parent.mkdir(parents=True)
            new_path.write_bytes(b"accepted")
            old_path.parent.mkdir(parents=True)
            old_path.write_bytes(b"late old attempt")
            self.assertEqual(new_path.read_bytes(), b"accepted")
            self.assertTrue(access("new").is_visible_artifact(new_path))
            with self.assertRaises(InvalidWorkflowInvocation):
                access("new").output_path("../escape.tiff")
            synchronous = adapter.create(
                replace(
                    self.invocation(),
                    input_access=InputAccessContext(
                        kind="shared_storage", state={"root": directory}
                    ),
                )
            )
            self.assertEqual(
                synchronous.output_path("outputs/heatmap.tiff"),
                Path(directory) / "outputs" / "heatmap.tiff",
            )

    async def test_cancelled_empaia_acquisition_closes_gateway(self) -> None:
        started = asyncio.Event()
        closed = asyncio.Event()

        class Gateway:
            async def materialize_input(self, _name, _destination):
                started.set()
                await asyncio.Event().wait()

            async def close(self):
                closed.set()
                raise OSError("cleanup failure")

        with TemporaryDirectory() as directory:
            adapter = EmpaiaInputAccessAdapter(
                lambda _context: Gateway(), Path(directory)
            )
            invocation = replace(
                self.invocation(),
                inputs={"slide": ProviderInput("slide")},
                input_access=InputAccessContext(
                    kind="empaia",
                    state={
                        "api_url": "https://example.invalid",
                        "job_id": "job",
                        "credential": "secret",
                        "processing_attempt_id": "attempt",
                    },
                ),
            )
            task = asyncio.create_task(adapter.create(invocation))
            await asyncio.wait_for(started.wait(), timeout=1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1)
            self.assertTrue(closed.is_set())

    async def test_worker_overrides_caller_attempt_and_isolates_results(self) -> None:
        with TemporaryDirectory() as directory:
            attempts = []

            class Workflow:
                async def run(self, context):
                    output = context.output_path("result.bin")
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(str(len(attempts)).encode())
                    return WorkflowResult({"artifact": SharedArtifact(str(output))})

            definition = WorkflowDefinition(
                "artifact", {}, {"artifact": WorkflowValueDefinition("shared_artifact")}
            )
            catalog = WorkflowCatalog(
                (definition,),
                configurations={
                    "artifact": WorkflowConfiguration(
                        validator=validate_empty_configuration
                    )
                },
            )
            application = WorkflowApplication(definition, Workflow())

            async def dispatch(invocation):
                attempts.append(invocation.input_access.state["processing_attempt_id"])
                return await application.run(invocation)

            repository = InMemoryExecutionRepository()
            provider = StandaloneProviderAdapter(catalog)
            scheduler = ExecutionScheduler(
                repository=repository,
                providers={provider.name: provider},
                known_workflows=catalog.identifiers,
                default_timeout=timedelta(minutes=1),
                maximum_timeout=timedelta(minutes=1),
            )
            worker = ExecutionWorker(
                repository=repository,
                providers={provider.name: provider},
                dispatch=InMemoryWorkflowDispatch({"artifact": dispatch}),
                owner="test-worker",
                lease_duration=timedelta(seconds=30),
            )
            invocation = replace(
                self.invocation(),
                input_access=InputAccessContext(
                    kind="shared_storage",
                    state={"root": directory, "processing_attempt_id": "caller-value"},
                ),
            )
            scheduled = []
            for key in ("first", "second"):
                scheduled.append(
                    await scheduler.schedule(
                        ScheduleRequest(
                            workflow_id="artifact",
                            provider="standalone",
                            body=invocation_to_dict(invocation),
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
            paths = [Path(item.result.outputs["artifact"].path) for item in results]
            self.assertNotEqual(paths[0], paths[1])
            self.assertEqual(paths[0].read_bytes(), b"1")
            self.assertEqual(paths[1].read_bytes(), b"2")
            self.assertEqual(len(set(attempts)), 2)
            self.assertNotIn("caller-value", attempts)


class EmptyWorkflow:
    async def run(self, _context):
        return WorkflowResult(outputs={})
