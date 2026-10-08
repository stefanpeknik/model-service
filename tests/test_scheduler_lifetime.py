"""Check scheduler construction, worker lifetime, shutdown, and restart."""

from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime, timedelta
from unittest.mock import patch

from execution_scheduler.contracts import ExecutionStatus, ScheduleRequest
from execution_scheduler.domain import RetryPolicy
from execution_scheduler.persistence.memory import InMemoryExecutionRepository
from execution_scheduler.scheduler import ExecutionScheduler
from tests.test_execution_scheduler import MutableClock, StandaloneProvider
from workflow_execution.contracts import WorkflowResult
from workflow_execution.dispatch import InMemoryWorkflowDispatch


class SchedulerLifetimeTests(unittest.IsolatedAsyncioTestCase):
    """Exercise scheduler construction, worker lifetime, shutdown, and restart."""

    def setUp(self) -> None:
        """Create isolated fixtures and clients for scheduler construction, worker lifetime, shutdown, and restart.

        Returns:
            None: No return value.
        """
        self.clock = MutableClock(datetime(2026, 9, 17, 8, tzinfo=UTC))
        self.repository = InMemoryExecutionRepository(
            clock=self.clock,
            processing_policy=RetryPolicy(2, timedelta(0), timedelta(0)),
        )
        self.provider = StandaloneProvider()

    def scheduler(self, dispatch=None, **settings) -> ExecutionScheduler:
        """Construct a scheduler over the shared test repository with selectable dispatch and timing settings.

        Args:
            dispatch: Workflow invocation boundary, or a selectable fixture handler in
                lifetime tests.
            **settings: Keyword arguments forwarded to the delegated operation or
                accepted by the fixture interface.

        Returns:
            ExecutionScheduler: Scheduler using the shared test repository and selected
                dispatch behavior.
        """
        return ExecutionScheduler(
            repository=self.repository,
            providers={self.provider.name: self.provider},
            dispatch=InMemoryWorkflowDispatch(
                {"prostate-heatmap": dispatch or self.completed_dispatch}
            ),
            known_workflows={"prostate-heatmap"},
            default_timeout=timedelta(hours=1),
            maximum_timeout=timedelta(hours=6),
            clock=self.clock,
            poll_interval=0.001,
            **settings,
        )

    @staticmethod
    def completed_dispatch(_invocation, _execution_context):
        """Return an empty successful result for scheduler lifetime checks.

        Args:
            _invocation: Unused named inputs, resolved configuration, public access
                state, and absolute workflow deadline; retained to match the interface.
            _execution_context: Unused private attempt identity and provider context
                kept separate from the invocation; retained to match the interface.

        Returns:
            WorkflowResult: Named inline outputs and/or references to completed artifact
                files.
        """
        return WorkflowResult(outputs={})

    async def schedule(self, scheduler, key="request-1"):
        """Submit minimal standalone fixture work under the selected replay key.

        Args:
            scheduler: Durable scheduler boundary or its live HTTP client, according to
                the declared type.
            key: Replay identity string or the 32-byte AES key, according to this
                method's signature.

        Returns:
            object: The delegated operation's result or controlled fixture value
                described above.
        """
        return await scheduler.schedule(
            ScheduleRequest(
                workflow_id="prostate-heatmap",
                provider="standalone",
                body={},
                idempotency_key=key,
            )
        )

    async def wait_succeeded(self, scheduler, execution_id):
        """Poll for successful completion within the one-second lifetime test budget.

        Args:
            scheduler: Durable scheduler boundary or its live HTTP client, according to
                the declared type.
            execution_id: Durable execution identifier, not an external provider job ID.

        Returns:
            None: No return value.
        """
        async with asyncio.timeout(1):
            while (
                await scheduler.get(execution_id)
            ).status is not ExecutionStatus.SUCCEEDED:
                await asyncio.sleep(0.001)

    async def test_construction_starts_no_tasks_and_running_completes_work(self):
        """Verify construction starts no tasks and running completes work.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        tasks_before = asyncio.all_tasks()
        scheduler = self.scheduler()
        admitted = await self.schedule(scheduler)
        self.assertEqual(asyncio.all_tasks(), tasks_before)
        self.assertEqual(admitted.execution.status, ExecutionStatus.QUEUED)

        async with scheduler.running():
            await self.wait_succeeded(scheduler, admitted.execution.id)

        self.assertEqual(asyncio.all_tasks(), tasks_before)

    async def test_shutdown_joins_dispatch_renewal_and_maintenance_without_cancelling_execution(
        self,
    ):
        """Verify shutdown joins dispatch renewal and maintenance without cancelling execution.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        tasks_before = asyncio.all_tasks()
        started = asyncio.Event()
        renewed = asyncio.Event()
        stopped = asyncio.Event()
        renew = self.repository.renew

        async def tracked_renew(*args, **kwargs):
            """Delegate lease renewal and signal that the worker renewed its active claim.

            Args:
                *args: Positional arguments forwarded to the delegated operation or
                    accepted by the fixture interface.
                **kwargs: Keyword arguments forwarded to the delegated operation or
                    accepted by the fixture interface.

            Returns:
                object: The delegated operation's result or controlled fixture value
                    described above.
            """
            result = await renew(*args, **kwargs)
            renewed.set()
            return result

        async def dispatch(_invocation, _execution_context):
            """Simulate the scenario computation, using events or injected failures to control completion and ownership races.

            Args:
                _invocation: Unused named inputs, resolved configuration, public access
                    state, and absolute workflow deadline; retained to match the
                    interface.
                _execution_context: Unused private attempt identity and provider context
                    kept separate from the invocation; retained to match the interface.

            Returns:
                None: No return value.
            """
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        scheduler = self.scheduler(dispatch, lease_duration=timedelta(milliseconds=30))
        admitted = await self.schedule(scheduler)
        with patch.object(self.repository, "renew", side_effect=tracked_renew):
            async with scheduler.running():
                await asyncio.wait_for(started.wait(), timeout=1)
                await asyncio.wait_for(renewed.wait(), timeout=1)

        self.assertTrue(stopped.is_set())
        self.assertEqual(asyncio.all_tasks(), tasks_before)
        durable = await self.repository.get(admitted.execution.id)
        self.assertEqual(durable.status, ExecutionStatus.RUNNING)
        self.assertIsNotNone(durable.fencing_token)
        self.clock.advance(timedelta(seconds=1))
        replacement = self.scheduler()
        async with replacement.running():
            await self.wait_succeeded(replacement, admitted.execution.id)
        recovered = await self.repository.get(admitted.execution.id)
        self.assertEqual(recovered.processing_attempts, 2)

    async def test_duplicate_start_is_rejected_and_scheduler_can_restart(self):
        """Verify duplicate start is rejected and scheduler can restart.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        scheduler = self.scheduler()
        async with scheduler.running():
            with self.assertRaisesRegex(RuntimeError, "already running"):
                async with scheduler.running():
                    self.fail("duplicate start was accepted")
            admitted = await self.schedule(scheduler)
            await self.wait_succeeded(scheduler, admitted.execution.id)

        admitted = await self.schedule(scheduler, key="after-restart")
        async with scheduler.running():
            await self.wait_succeeded(scheduler, admitted.execution.id)

    async def test_queued_deadlines_advance_while_managed_dispatch_is_blocked(self):
        """Verify queued deadlines advance while managed dispatch is blocked.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        started = asyncio.Event()

        async def dispatch(_invocation, _execution_context):
            """Simulate the scenario computation, using events or injected failures to control completion and ownership races.

            Args:
                _invocation: Unused named inputs, resolved configuration, public access
                    state, and absolute workflow deadline; retained to match the
                    interface.
                _execution_context: Unused private attempt identity and provider context
                    kept separate from the invocation; retained to match the interface.

            Returns:
                None: No return value.
            """
            started.set()
            await asyncio.Event().wait()

        scheduler = self.scheduler(dispatch)
        await self.schedule(scheduler)
        async with scheduler.running():
            await asyncio.wait_for(started.wait(), timeout=1)
            queued = await scheduler.schedule(
                ScheduleRequest(
                    workflow_id="prostate-heatmap",
                    provider="standalone",
                    body={},
                    idempotency_key="queued-deadline",
                    requested_timeout_seconds=1,
                )
            )
            self.clock.advance(timedelta(seconds=2))
            async with asyncio.timeout(1):
                while (
                    await scheduler.get(queued.execution.id)
                ).status is not ExecutionStatus.FAILED:
                    await asyncio.sleep(0.001)
            expired = await scheduler.get(queued.execution.id)
            self.assertIsNotNone(expired.error)
            self.assertEqual(expired.error.code, "execution_deadline_exceeded")

    async def test_cancelling_owner_scope_joins_tasks_and_preserves_durable_work(self):
        """Verify cancelling owner scope joins tasks and preserves durable work.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        tasks_before = asyncio.all_tasks()
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def dispatch(_invocation, _execution_context):
            """Simulate the scenario computation, using events or injected failures to control completion and ownership races.

            Args:
                _invocation: Unused named inputs, resolved configuration, public access
                    state, and absolute workflow deadline; retained to match the
                    interface.
                _execution_context: Unused private attempt identity and provider context
                    kept separate from the invocation; retained to match the interface.

            Returns:
                None: No return value.
            """
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

        scheduler = self.scheduler(dispatch)
        admitted = await self.schedule(scheduler)

        async def serve():
            """Keep the scheduler lifetime open until its owning task is cancelled.

            Returns:
                None: No return value.
            """
            async with scheduler.running():
                await asyncio.Event().wait()

        owner = asyncio.create_task(serve())
        await asyncio.wait_for(started.wait(), timeout=1)
        owner.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await owner
        self.assertTrue(stopped.is_set())
        self.assertEqual(asyncio.all_tasks(), tasks_before)
        self.assertEqual(
            (await scheduler.get(admitted.execution.id)).status,
            ExecutionStatus.RUNNING,
        )

    async def test_worker_failure_is_surfaced_and_lifetime_can_be_reentered(self):
        """Verify worker failure is surfaced and lifetime can be reentered.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        scheduler = self.scheduler()
        with (
            patch(
                "execution_scheduler.worker.ExecutionWorker.run",
                side_effect=RuntimeError("worker failed"),
            ),
            self.assertRaises(ExceptionGroup) as raised,
        ):
            async with scheduler.running():
                await asyncio.Event().wait()
        self.assertIsInstance(raised.exception.exceptions[0], RuntimeError)
        admitted = await self.schedule(scheduler)
        async with scheduler.running():
            await self.wait_succeeded(scheduler, admitted.execution.id)

    def test_nonpositive_timing_is_rejected_before_startup(self):
        """Verify nonpositive timing is rejected before startup.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        with self.assertRaisesRegex(ValueError, "lease_duration"):
            self.scheduler(lease_duration=timedelta(0))
        with self.assertRaisesRegex(ValueError, "poll_interval"):
            ExecutionScheduler(
                repository=self.repository,
                providers={},
                dispatch=InMemoryWorkflowDispatch({}),
                known_workflows=set(),
                default_timeout=timedelta(hours=1),
                maximum_timeout=timedelta(hours=6),
                poll_interval=0,
            )
