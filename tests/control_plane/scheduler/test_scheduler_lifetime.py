"""Check scheduler construction, worker lifetime, shutdown, and restart."""

from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

from model_service.contracts.result import RunResult
from model_service.control_plane.scheduler.execution_scheduler import ExecutionScheduler
from model_service.control_plane.scheduler.internal_state import (
    RenewDirective,
    RetryPolicy,
)
from model_service.control_plane.scheduler.public import AnalysisScheduleRequest
from model_service.control_plane.scheduler.status import ExecutionStatus
from tests.support.memory_repository import FakeDispatch, InMemoryExecutionRepository
from tests.support.scheduling import (
    MutableClock,
    StandaloneProvider,
    resolve_test_analysis,
)


if TYPE_CHECKING:
    from model_service.contracts.invocation import ExecutionContext, Invocation
    from model_service.control_plane.scheduler.public import ScheduleResult
    from tests.support.memory_repository import Handler


class SchedulerLifetimeTests(unittest.IsolatedAsyncioTestCase):
    """Exercise scheduler construction, worker lifetime, shutdown, and restart."""

    def setUp(self) -> None:
        """Create isolated fixtures and clients for scheduler construction, worker lifetime, shutdown, and restart."""
        self.clock = MutableClock(datetime(2026, 9, 17, 8, tzinfo=UTC))
        self.repository = InMemoryExecutionRepository(
            clock=self.clock,
            processing_policy=RetryPolicy(2, timedelta(0), timedelta(0)),
        )
        self.provider = StandaloneProvider()

    def scheduler(
        self, dispatch: Handler | None = None, **settings: Any
    ) -> ExecutionScheduler:
        """Construct a scheduler over the shared test repository with selectable dispatch and timing settings."""
        return ExecutionScheduler(
            repository=self.repository,
            providers={self.provider.name: self.provider},
            dispatch=FakeDispatch({"a1": dispatch or self.completed_dispatch}),
            resolve_analysis=resolve_test_analysis,
            default_timeout=timedelta(hours=1),
            maximum_timeout=timedelta(hours=6),
            clock=self.clock,
            poll_interval=0.001,
            **settings,
        )

    @staticmethod
    def completed_dispatch(
        _invocation: Invocation, _execution_context: ExecutionContext
    ) -> RunResult:
        """Return an empty successful result for scheduler lifetime checks."""
        return RunResult(outputs={})

    async def schedule(
        self, scheduler: ExecutionScheduler, key: str = "request-1"
    ) -> ScheduleResult:
        """Submit minimal standalone fixture work under the selected replay key."""
        return await scheduler.schedule(
            AnalysisScheduleRequest(
                analysis_id="a1",
                provider="standalone",
                body={},
                idempotency_key=key,
            )
        )

    async def wait_succeeded(
        self, scheduler: ExecutionScheduler, execution_id: str
    ) -> None:
        """Poll for successful completion within the one-second lifetime test budget."""
        async with asyncio.timeout(1):
            while (
                await scheduler.get(execution_id)
            ).status is not ExecutionStatus.SUCCEEDED:
                await asyncio.sleep(0.001)

    async def test_construction_starts_no_tasks_and_running_completes_work(
        self,
    ) -> None:
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
    ) -> None:
        tasks_before = asyncio.all_tasks()
        started = asyncio.Event()
        renewed = asyncio.Event()
        stopped = asyncio.Event()
        renew = self.repository.renew

        async def tracked_renew(*args: Any, **kwargs: Any) -> RenewDirective:
            """Delegate lease renewal and signal that the worker renewed its active claim."""
            result = await renew(*args, **kwargs)
            renewed.set()
            return result

        async def dispatch(
            _invocation: Invocation, _execution_context: ExecutionContext
        ) -> RunResult:
            """Simulate the scenario computation, using events or injected failures to control completion and ownership races."""
            started.set()
            try:
                await asyncio.Event().wait()
                raise AssertionError("the event is never set")
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

    async def test_duplicate_start_is_rejected_and_scheduler_can_restart(self) -> None:
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

    async def test_queued_deadlines_advance_while_managed_dispatch_is_blocked(
        self,
    ) -> None:
        started = asyncio.Event()

        async def dispatch(
            _invocation: Invocation, _execution_context: ExecutionContext
        ) -> RunResult:
            """Simulate the scenario computation, using events or injected failures to control completion and ownership races."""
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("the event is never set")

        scheduler = self.scheduler(dispatch)
        await self.schedule(scheduler)
        async with scheduler.running():
            await asyncio.wait_for(started.wait(), timeout=1)
            queued = await scheduler.schedule(
                AnalysisScheduleRequest(
                    analysis_id="a1",
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
            assert expired.error is not None
            self.assertEqual(expired.error.code, "execution_deadline_exceeded")

    async def test_cancelling_owner_scope_joins_tasks_and_preserves_durable_work(
        self,
    ) -> None:
        tasks_before = asyncio.all_tasks()
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def dispatch(
            _invocation: Invocation, _execution_context: ExecutionContext
        ) -> RunResult:
            """Simulate the scenario computation, using events or injected failures to control completion and ownership races."""
            started.set()
            try:
                await asyncio.Event().wait()
                raise AssertionError("the event is never set")
            finally:
                stopped.set()

        scheduler = self.scheduler(dispatch)
        admitted = await self.schedule(scheduler)

        async def serve() -> None:
            """Keep the scheduler lifetime open until its owning task is cancelled."""
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

    async def test_worker_failure_is_surfaced_and_lifetime_can_be_reentered(
        self,
    ) -> None:
        scheduler = self.scheduler()
        with (
            patch(
                "model_service.control_plane.scheduler.worker.ExecutionWorker.run",
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
