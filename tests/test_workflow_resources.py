"""Check deadline-aware acquisition, bounded cleanup, repeated cancellation, and attempt-isolated artifacts."""

from __future__ import annotations

import asyncio
import gc
import unittest
import weakref
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from execution_scheduler.active_work import ActiveWorkRegistry
from execution_scheduler.contracts import ExecutionStatus, ScheduleRequest
from execution_scheduler.persistence.memory import InMemoryExecutionRepository
from execution_scheduler.scheduler import ExecutionScheduler
from execution_scheduler.worker import ExecutionWorker
from integrations.empaia.input_access import EmpaiaInputAccessAdapter
from integrations.shared_storage import SharedStorageAccessAdapter
from integrations.standalone import StandaloneProviderAdapter
from workflow_execution.application import WorkflowApplication
from workflow_execution.catalog import (
    WorkflowCatalog,
    WorkflowConfiguration,
    WorkflowDefinition,
    WorkflowValueDefinition,
    validate_empty_configuration,
)
from workflow_execution.cleanup import close_resource
from workflow_execution.contracts import (
    InputAccessContext,
    InvalidWorkflowInvocation,
    ProviderInput,
    SharedArtifact,
    WorkflowExecutionContext,
    WorkflowFailure,
    WorkflowInvocation,
    WorkflowResult,
)
from workflow_execution.dispatch import InMemoryWorkflowDispatch


class WorkflowResourceTests(unittest.IsolatedAsyncioTestCase):
    """Exercise deadline-aware acquisition, bounded cleanup, repeated cancellation, and attempt-isolated artifacts."""

    @staticmethod
    def invocation() -> WorkflowInvocation:
        """Build an invocation with the test access kind/state and a future absolute deadline.

        Returns:
            WorkflowInvocation: Typed inputs, resolved configuration, access settings,
                and absolute deadline.
        """
        return WorkflowInvocation(
            inputs={},
            resolved_configuration={},
            input_access=InputAccessContext(kind="test", state={}),
            deadline_at=datetime.now(UTC) + timedelta(minutes=1),
        )

    async def test_deadline_includes_input_acquisition(self) -> None:
        """Verify deadline includes input acquisition.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        cancelled = asyncio.Event()

        class Adapter:
            """Provide the adapter fixture local to this failure or cancellation scenario."""

            async def create(self, _invocation, _execution_context):
                """Construct or block acquisition of the scenario access fixture without contacting an external provider.

                Args:
                    _invocation: Unused named inputs, resolved configuration, public
                        access state, and absolute workflow deadline; retained to match
                        the interface.
                    _execution_context: Unused private attempt identity and provider
                        context kept separate from the invocation; retained to match the
                        interface.

                Returns:
                    None: No return value.
                """
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
        """Verify cleanup preserves primary failure.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """

        class Workflow:
            """Provide the workflow fixture local to this failure or cancellation scenario."""

            async def run(self, _context):
                """Execute the scenario workflow fixture to expose the result or failure checked by its enclosing test.

                Args:
                    _context: Unused workflow access/configuration or private provider
                        job context, according to the declared type; retained to match
                        the interface.

                Returns:
                    None: No return value.
                """
                raise primary

        class Access:
            """Provide the access fixture local to this failure or cancellation scenario."""

            async def aclose(self):
                """Simulate the scenario cleanup outcome and signal resource shutdown through test events.

                Returns:
                    None: No return value.

                Raises:
                    OSError: Input, output, or native storage access fails.
                """
                raise OSError("cleanup failure")

        class Adapter:
            """Provide the adapter fixture local to this failure or cancellation scenario."""

            def create(self, _invocation, _execution_context):
                """Construct or block acquisition of the scenario access fixture without contacting an external provider.

                Args:
                    _invocation: Unused named inputs, resolved configuration, public
                        access state, and absolute workflow deadline; retained to match
                        the interface.
                    _execution_context: Unused private attempt identity and provider
                        context kept separate from the invocation; retained to match the
                        interface.

                Returns:
                    object: The delegated operation's result or controlled fixture value
                        described above.
                """
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
        """Verify repeated cancellation waits for cleanup.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        started = asyncio.Event()
        closing = asyncio.Event()
        release_close = asyncio.Event()
        closed = asyncio.Event()

        class Workflow:
            """Provide the workflow fixture local to this failure or cancellation scenario."""

            async def run(self, _context):
                """Execute the scenario workflow fixture to expose the result or failure checked by its enclosing test.

                Args:
                    _context: Unused workflow access/configuration or private provider
                        job context, according to the declared type; retained to match
                        the interface.

                Returns:
                    None: No return value.
                """
                started.set()
                await asyncio.Event().wait()

        class Access:
            """Provide the access fixture local to this failure or cancellation scenario."""

            async def aclose(self):
                """Simulate the scenario cleanup outcome and signal resource shutdown through test events.

                Returns:
                    None: No return value.
                """
                closing.set()
                await release_close.wait()
                closed.set()

        class Adapter:
            """Provide the adapter fixture local to this failure or cancellation scenario."""

            def create(self, _invocation, _execution_context):
                """Construct or block acquisition of the scenario access fixture without contacting an external provider.

                Args:
                    _invocation: Unused named inputs, resolved configuration, public
                        access state, and absolute workflow deadline; retained to match
                        the interface.
                    _execution_context: Unused private attempt identity and provider
                        context kept separate from the invocation; retained to match the
                        interface.

                Returns:
                    object: The delegated operation's result or controlled fixture value
                        described above.
                """
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
        """Verify cleanup is bounded.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        close_cancelled = asyncio.Event()

        class Access:
            """Provide the access fixture local to this failure or cancellation scenario."""

            async def aclose(self):
                """Simulate the scenario cleanup outcome and signal resource shutdown through test events.

                Returns:
                    None: No return value.
                """
                try:
                    await asyncio.Event().wait()
                finally:
                    close_cancelled.set()

        class Adapter:
            """Provide the adapter fixture local to this failure or cancellation scenario."""

            def create(self, _invocation, _execution_context):
                """Construct or block acquisition of the scenario access fixture without contacting an external provider.

                Args:
                    _invocation: Unused named inputs, resolved configuration, public
                        access state, and absolute workflow deadline; retained to match
                        the interface.
                    _execution_context: Unused private attempt identity and provider
                        context kept separate from the invocation; retained to match the
                        interface.

                Returns:
                    object: The delegated operation's result or controlled fixture value
                        described above.
                """
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
        """Verify resource self cancellation is a cleanup failure.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """

        class Access:
            """Provide the access fixture local to this failure or cancellation scenario."""

            async def aclose(self):
                """Simulate the scenario cleanup outcome and signal resource shutdown through test events.

                Returns:
                    None: No return value.

                Raises:
                    asyncio.CancelledError: The owning task, caller, deadline, or lease
                        monitor cancels the operation.
                """
                raise asyncio.CancelledError

        class Adapter:
            """Provide the adapter fixture local to this failure or cancellation scenario."""

            def create(self, _invocation, _execution_context):
                """Construct or block acquisition of the scenario access fixture without contacting an external provider.

                Args:
                    _invocation: Unused named inputs, resolved configuration, public
                        access state, and absolute workflow deadline; retained to match
                        the interface.
                    _execution_context: Unused private attempt identity and provider
                        context kept separate from the invocation; retained to match the
                        interface.

                Returns:
                    object: The delegated operation's result or controlled fixture value
                        described above.
                """
                return Access()

        class Workflow:
            """Provide the workflow fixture local to this failure or cancellation scenario."""

            async def run(self, _context):
                """Execute the scenario workflow fixture to expose the result or failure checked by its enclosing test.

                Args:
                    _context: Unused workflow access/configuration or private provider
                        job context, according to the declared type; retained to match
                        the interface.

                Returns:
                    None: No return value.

                Raises:
                    InvalidWorkflowInvocation: Inputs, resolved configuration, access
                        context, or deadline violate the workflow contract.
                """
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
        """Verify timed out cleanup remains owned until it stops.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        cancellation_received = asyncio.Event()
        closed = asyncio.Event()
        task_ref = None

        async def close():
            """Simulate cancellation-resistant cleanup and expose its task lifetime through test events and a weak reference.

            Returns:
                None: No return value.
            """
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
        """Verify scheduled artifacts are isolated by attempt.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        with TemporaryDirectory() as directory:
            adapter = SharedStorageAccessAdapter()

            def access(attempt):
                """Construct shared-storage access with the supplied attempt ID to compare isolated output paths.

                Args:
                    attempt: One-based retry attempt number, or the fixture attempt
                        identity when constructing access.

                Returns:
                    object: The delegated operation's result or controlled fixture value
                        described above.
                """
                return adapter.create(
                    replace(
                        self.invocation(),
                        input_access=InputAccessContext(
                            kind="shared_storage",
                            state={"root": directory},
                        ),
                    ),
                    WorkflowExecutionContext(processing_attempt_id=attempt),
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
                ),
                WorkflowExecutionContext(),
            )
            self.assertEqual(
                synchronous.output_path("outputs/heatmap.tiff"),
                Path(directory) / "outputs" / "heatmap.tiff",
            )

    async def test_cancelled_empaia_acquisition_closes_gateway(self) -> None:
        """Verify cancelled EMPAIA acquisition closes gateway.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        started = asyncio.Event()
        closed = asyncio.Event()

        class Gateway:
            """Provide the gateway fixture local to this failure or cancellation scenario."""

            async def materialize_input(self, _name, _destination):
                """Block fake provider acquisition after signalling that materialization started.

                Args:
                    _name: Unused input/output name or configuration field selected by
                        this operation; retained to match the interface.
                    _destination: Unused local path where input materialization or mask
                        writing must produce a file; retained to match the interface.

                Returns:
                    None: No return value.
                """
                started.set()
                await asyncio.Event().wait()

            async def close(self):
                """Simulate cancellation-resistant cleanup and expose its task lifetime through test events and a weak reference.

                Returns:
                    None: No return value.

                Raises:
                    OSError: Input, output, or native storage access fails.
                """
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
                    },
                ),
            )
            task = asyncio.create_task(
                adapter.create(
                    invocation,
                    WorkflowExecutionContext(
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

    async def test_worker_supplies_runtime_attempt_and_isolates_results(self) -> None:
        """Verify worker supplies runtime attempt and isolates results.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        with TemporaryDirectory() as directory:
            attempts = []

            class Workflow:
                """Provide the workflow fixture local to this failure or cancellation scenario."""

                async def run(self, context):
                    """Execute the scenario workflow fixture to expose the result or failure checked by its enclosing test.

                    Args:
                        context: Workflow access/configuration or private provider job
                            context, according to the declared type.

                    Returns:
                        WorkflowResult: Named inline outputs and/or references to
                            completed artifact files.
                    """
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
            application = WorkflowApplication(definition, Workflow(), catalog=catalog)

            async def dispatch(invocation, execution_context):
                """Simulate the scenario computation, using events or injected failures to control completion and ownership races.

                Args:
                    invocation: Named inputs, resolved configuration, public access
                        state, and absolute workflow deadline.
                    execution_context: Private attempt identity and provider context
                        kept separate from the invocation.

                Returns:
                    object: The delegated operation's result or controlled fixture value
                        described above.
                """
                attempts.append(execution_context.processing_attempt_id)
                self.assertEqual(invocation.input_access.state, {"root": directory})
                return await application.run(invocation, execution_context)

            repository = InMemoryExecutionRepository()
            provider = StandaloneProviderAdapter(catalog)
            scheduler = ExecutionScheduler(
                repository=repository,
                dispatch=InMemoryWorkflowDispatch({}),
                providers={provider.name: provider},
                known_workflows=catalog.identifiers,
                default_timeout=timedelta(minutes=1),
                maximum_timeout=timedelta(minutes=1),
            )
            worker = ExecutionWorker(
                repository=repository,
                active_work=ActiveWorkRegistry(),
                providers={provider.name: provider},
                dispatch=InMemoryWorkflowDispatch({"artifact": dispatch}),
                owner="test-worker",
                lease_duration=timedelta(seconds=30),
            )
            scheduled = []
            for key in ("first", "second"):
                scheduled.append(
                    await scheduler.schedule(
                        ScheduleRequest(
                            workflow_id="artifact",
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
            paths = [Path(item.result.outputs["artifact"].path) for item in results]
            self.assertNotEqual(paths[0], paths[1])
            self.assertEqual(paths[0].read_bytes(), b"1")
            self.assertEqual(paths[1].read_bytes(), b"2")
            self.assertEqual(len(set(attempts)), 2)
            self.assertNotIn(None, attempts)


class EmptyWorkflow:
    """Return no outputs for resource lifetime tests that need no computation."""

    async def run(self, _context):
        """Return an empty successful result so tests can isolate resource lifetime behavior.

        Args:
            _context: Unused workflow access/configuration or private provider job
                context, according to the declared type; retained to match the
                interface.

        Returns:
            WorkflowResult: Named inline outputs and/or references to completed artifact
                files.
        """
        return WorkflowResult(outputs={})
