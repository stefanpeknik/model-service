"""Check admission, leases, recovery, retries, cancellation, and provider finish."""

from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from execution_scheduler.active_work import ActiveWorkRegistry
from execution_scheduler.contracts import (
    ExecutionStatus,
    InvalidScheduleRequestError,
    ScheduleConflictError,
    ScheduleRequest,
    SchedulingUnavailableError,
)
from execution_scheduler.domain import (
    Admission,
    PreparedWorkflow,
    ProcessingSucceeded,
    ProviderFinishSucceeded,
    RenewDirective,
    RetryPolicy,
    StaleWorkClaimError,
    SubmissionIdentity,
    WorkKind,
)
from execution_scheduler.persistence.memory import InMemoryExecutionRepository
from execution_scheduler.providers import (
    ProviderCancellation,
    ProviderFailure,
    ProviderFinishFailureError,
    ProviderSuccess,
)
from execution_scheduler.repository import (
    ExecutionRepositoryUnavailableError,
)
from execution_scheduler.scheduler import ExecutionScheduler
from execution_scheduler.worker import ExecutionWorker
from integrations.standalone import StandaloneProviderAdapter
from workflow_execution.catalog import WorkflowCatalog
from workflow_execution.contracts import (
    InputAccessContext,
    SharedArtifact,
    WorkflowExecutionContext,
    WorkflowInvocation,
    WorkflowResult,
)
from workflow_execution.dispatch import (
    ConfirmedDispatchRejectionError,
    InMemoryWorkflowDispatch,
    RemoteWorkflowFailureError,
    UnknownDispatchOutcomeError,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from execution_scheduler.providers import BeforeMutation, ProviderFinishOutcome
    from workflow_execution.contracts import JsonValue


class StandaloneProvider:
    """Prepare minimal scheduler invocations without shared-file checks so tests can isolate orchestration."""

    name = "standalone"

    def identify(self, request: ScheduleRequest) -> SubmissionIdentity:
        """Use the test replay key with a fixed fingerprint to isolate scheduler replay behavior.

        Args:
            request: Provider submission with workflow ID, caller body, replay key, and
                optional execution budget.

        Returns:
            SubmissionIdentity: Provider-scoped replay key and canonical request
                fingerprint.
        """
        assert request.idempotency_key is not None
        return SubmissionIdentity(
            provider=self.name,
            key=request.idempotency_key,
            fingerprint="same-request",
        )

    async def prepare(
        self,
        request: ScheduleRequest,
        deadline_at: datetime,
    ) -> PreparedWorkflow:
        """Create an empty shared-storage invocation with the scheduler-supplied deadline.

        Args:
            request: Provider submission with workflow ID, caller body, replay key, and
                optional execution budget.
            deadline_at: Absolute aware deadline for the entire execution, including
                preparation.

        Returns:
            PreparedWorkflow: Validated invocation, private provider state, and
                applicable deadline limits.
        """
        return PreparedWorkflow(
            invocation=WorkflowInvocation(
                inputs={},
                resolved_configuration={},
                input_access=InputAccessContext(
                    kind="shared_storage",
                    state={"root": "/shared"},
                ),
                deadline_at=deadline_at,
            ),
            protected_context={},
        )

    async def finish(
        self,
        _protected_context: dict[str, JsonValue],
        _outcome: ProviderFinishOutcome,
        *,
        before_mutation: BeforeMutation,
    ) -> None:
        """Complete without external writes or mutation guarding for standalone orchestration tests.

        Args:
            _protected_context: Unused private provider state that persistence encrypts
                and internal dispatch carries separately; retained to match the
                interface.
            _outcome: Unused stored success, safe failure, or cancellation to report
                through the provider; retained to match the interface.
            before_mutation: Async guard awaited before the first external write to
                persist and validate its durable claim.

        Returns:
            None: No return value.
        """
        del before_mutation


class RecordingProvider(StandaloneProvider):
    """Record provider outcomes and inject finish failures independently of workflow computation."""

    def __init__(
        self,
        *,
        protected_context: dict[str, JsonValue] | None = None,
        finish_failures: list[ProviderFinishFailureError] | None = None,
    ) -> None:
        """Set protected context, queued finish failures, and an empty recorded-outcome list.

        Args:
            protected_context: Private provider state that persistence encrypts and
                internal dispatch carries separately.
            finish_failures: Optional queued failures injected into successive provider
                finish calls.
        """
        self.protected_context = protected_context or {}
        self.finish_failures = finish_failures or []
        self.outcomes: list[ProviderFinishOutcome] = []

    async def prepare(
        self,
        request: ScheduleRequest,
        deadline_at: datetime,
    ) -> PreparedWorkflow:
        """Attach configured protected context to the minimal prepared invocation.

        Args:
            request: Provider submission with workflow ID, caller body, replay key, and
                optional execution budget.
            deadline_at: Absolute aware deadline for the entire execution, including
                preparation.

        Returns:
            PreparedWorkflow: Validated invocation, private provider state, and
                applicable deadline limits.
        """
        prepared = await super().prepare(request, deadline_at)
        return PreparedWorkflow(
            invocation=prepared.invocation,
            protected_context=self.protected_context,
        )

    async def finish(
        self,
        _protected_context: dict[str, JsonValue],
        outcome: ProviderFinishOutcome,
        *,
        before_mutation: BeforeMutation,
    ) -> None:
        """Record the outcome and raise the next injected finish failure if configured.

        Args:
            _protected_context: Unused private provider state that persistence encrypts
                and internal dispatch carries separately; retained to match the
                interface.
            outcome: Stored success, safe failure, or cancellation to report through the
                provider.
            before_mutation: Async guard awaited before the first external write to
                persist and validate its durable claim.

        Returns:
            None: No return value.
        """
        del before_mutation
        self.outcomes.append(outcome)
        if self.finish_failures:
            raise self.finish_failures.pop(0)


class MutableClock:
    """Provide a manually advanced timestamp for scheduler lease, retry, and deadline tests."""

    def __init__(self, now: datetime) -> None:
        """Set the initial aware timestamp supplied by the test.

        Args:
            now: Aware current time supplied by the repository or validation clock.
        """
        self.now = now

    def __call__(self) -> datetime:
        """Return the current test timestamp without advancing it.

        Returns:
            datetime: Timezone-aware timestamp used for deadlines or durable lifecycle
                decisions.
        """
        return self.now

    def advance(self, duration: timedelta) -> None:
        """Advance simulated time by the supplied duration without sleeping.

        Args:
            duration: Amount of time to advance or wait in a deterministic lifecycle
                scenario.

        Returns:
            None: No return value.
        """
        self.now += duration


class UnavailableExecutionRepository(InMemoryExecutionRepository):
    """Inject replay-lookup storage unavailability into scheduler admission."""

    async def find_submission(
        self,
        _identity: SubmissionIdentity,
    ) -> None:
        """Raise storage unavailability during the early replay lookup.

        Args:
            _identity: Unused provider-scoped replay key and request fingerprint;
                retained to match the interface.

        Returns:
            None: Matching internal record, or None if the provider/key pair has never
                been admitted.

        Raises:
            ExecutionRepositoryUnavailableError: Database access or the storage
                transaction fails.
        """
        raise ExecutionRepositoryUnavailableError("database unavailable")


class ExecutionSchedulerTests(unittest.IsolatedAsyncioTestCase):
    """Exercise admission, leases, recovery, retries, cancellation, and provider finish."""

    async def test_claim_lease_loss_returns_before_another_worker_recovers_it(
        self,
    ) -> None:
        """Verify claim lease loss returns before another worker recovers it.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        clock = MutableClock(datetime(2026, 9, 17, 8, tzinfo=UTC))

        async def dispatch(
            _invocation: WorkflowInvocation,
            _execution_context: WorkflowExecutionContext,
        ) -> WorkflowResult:
            """Simulate the scenario computation, using events or injected failures to control completion and ownership races.

            Args:
                _invocation: Unused named inputs, resolved configuration, public access
                    state, and absolute workflow deadline; retained to match the
                    interface.
                _execution_context: Unused private attempt identity and provider context
                    kept separate from the invocation; retained to match the interface.

            Returns:
                WorkflowResult: Named inline outputs and/or references to completed
                    artifact files.

            Raises:
                AssertionError: An expected invariant or fixture interaction does not
                    hold.
            """
            clock.advance(timedelta(seconds=2))
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        repository = InMemoryExecutionRepository(clock=clock)
        scheduler, worker = self._scheduler_and_worker(
            repository,
            StandaloneProvider(),
            InMemoryWorkflowDispatch({"prostate-heatmap": dispatch}),
            clock=clock,
            lease_duration=timedelta(milliseconds=30),
        )
        admitted = await scheduler.schedule(self._request())

        self.assertTrue(await asyncio.wait_for(worker.run_once(), timeout=1))
        durable = await repository.get(admitted.execution.id)
        self.assertEqual(durable.status, "running")
        self.assertIsNotNone(durable.fencing_token)
        self.assertIsNone(durable.result)

    async def test_worker_shutdown_cancels_active_dispatch_and_propagates(self) -> None:
        """Verify worker shutdown cancels active dispatch and propagates.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def dispatch(
            _invocation: WorkflowInvocation,
            _execution_context: WorkflowExecutionContext,
        ) -> WorkflowResult:
            """Simulate the scenario computation, using events or injected failures to control completion and ownership races.

            Args:
                _invocation: Unused named inputs, resolved configuration, public access
                    state, and absolute workflow deadline; retained to match the
                    interface.
                _execution_context: Unused private attempt identity and provider context
                    kept separate from the invocation; retained to match the interface.

            Returns:
                WorkflowResult: Named inline outputs and/or references to completed
                    artifact files.

            Raises:
                AssertionError: An expected invariant or fixture interaction does not
                    hold.
            """
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
            raise AssertionError("unreachable")

        scheduler, worker = self._scheduler_and_worker(
            InMemoryExecutionRepository(),
            StandaloneProvider(),
            InMemoryWorkflowDispatch({"prostate-heatmap": dispatch}),
        )
        await scheduler.schedule(self._request())
        worker_task = self._start_worker(worker)
        await asyncio.wait_for(started.wait(), timeout=1)

        worker_task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(worker_task, timeout=1)
        self.assertTrue(stopped.is_set())

    async def test_renewal_error_stops_provider_finish(self) -> None:
        """Verify renewal error stops provider finish.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        stopped = asyncio.Event()

        class FailingRenewalRepository(InMemoryExecutionRepository):
            """Provide the failing renewal repository fixture local to this failure or cancellation scenario."""

            async def renew(
                self,
                execution_id: str,
                fencing_token: str,
                lease_duration: timedelta,
            ) -> RenewDirective:
                """Inject a lease-renewal failure or stall to verify work stops when ownership cannot be maintained.

                Args:
                    execution_id: Durable execution identifier, not an external provider
                        job ID.
                    fencing_token: Current claim token; stale tokens cannot mutate
                        execution state.
                    lease_duration: How long a claimed attempt retains ownership without
                        renewal.

                Returns:
                    RenewDirective: Instruction to continue or stop the current attempt.

                Raises:
                    ExecutionRepositoryUnavailableError: Database access or the storage
                        transaction fails.
                """
                raise ExecutionRepositoryUnavailableError("renewal unavailable")

        class BlockingProvider(StandaloneProvider):
            """Provide the blocking provider fixture local to this failure or cancellation scenario."""

            async def finish(
                self,
                _protected_context: dict[str, JsonValue],
                _outcome: ProviderFinishOutcome,
                *,
                before_mutation: BeforeMutation,
            ) -> None:
                """Block fake provider completion until cancellation exposes lost lease ownership.

                Args:
                    _protected_context: Unused private provider state that persistence
                        encrypts and internal dispatch carries separately; retained to
                        match the interface.
                    _outcome: Unused stored success, safe failure, or cancellation to
                        report through the provider; retained to match the interface.
                    before_mutation: Async guard awaited before the first external write
                        to persist and validate its durable claim.

                Returns:
                    None: No return value.
                """
                try:
                    await asyncio.Event().wait()
                finally:
                    stopped.set()

        repository = FailingRenewalRepository()
        scheduler, worker = self._scheduler_and_worker(
            repository,
            BlockingProvider(),
            InMemoryWorkflowDispatch(
                {
                    "prostate-heatmap": lambda _invocation, _execution_context: (
                        WorkflowResult(outputs={})
                    )
                }
            ),
            lease_duration=timedelta(milliseconds=30),
        )
        admitted = await scheduler.schedule(self._request())
        self.assertTrue(await worker.run_once())

        with self.assertRaises(ExecutionRepositoryUnavailableError):
            await asyncio.wait_for(worker.run_once(), timeout=1)
        self.assertTrue(stopped.is_set())
        durable = await repository.get(admitted.execution.id)
        self.assertEqual(durable.status, "running")
        self.assertEqual(durable.work_kind, WorkKind.PROVIDER_FINISH)
        self.assertFalse(durable.finish_mutation_started)

    async def test_stalled_renewal_stops_dispatch_before_abandoning_heartbeats(
        self,
    ) -> None:
        """Verify stalled renewal stops dispatch before abandoning heartbeats.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        stopped = asyncio.Event()

        class StalledRenewalRepository(InMemoryExecutionRepository):
            """Provide the stalled renewal repository fixture local to this failure or cancellation scenario."""

            async def renew(
                self,
                execution_id: str,
                fencing_token: str,
                lease_duration: timedelta,
            ) -> RenewDirective:
                """Inject a lease-renewal failure or stall to verify work stops when ownership cannot be maintained.

                Args:
                    execution_id: Durable execution identifier, not an external provider
                        job ID.
                    fencing_token: Current claim token; stale tokens cannot mutate
                        execution state.
                    lease_duration: How long a claimed attempt retains ownership without
                        renewal.

                Returns:
                    RenewDirective: Instruction to continue or stop the current attempt.

                Raises:
                    AssertionError: An expected invariant or fixture interaction does
                        not hold.
                """
                await asyncio.Event().wait()
                raise AssertionError("unreachable")

        async def dispatch(
            _invocation: WorkflowInvocation,
            _execution_context: WorkflowExecutionContext,
        ) -> WorkflowResult:
            """Simulate the scenario computation, using events or injected failures to control completion and ownership races.

            Args:
                _invocation: Unused named inputs, resolved configuration, public access
                    state, and absolute workflow deadline; retained to match the
                    interface.
                _execution_context: Unused private attempt identity and provider context
                    kept separate from the invocation; retained to match the interface.

            Returns:
                WorkflowResult: Named inline outputs and/or references to completed
                    artifact files.

            Raises:
                AssertionError: An expected invariant or fixture interaction does not
                    hold.
            """
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
            raise AssertionError("unreachable")

        scheduler, worker = self._scheduler_and_worker(
            StalledRenewalRepository(),
            StandaloneProvider(),
            InMemoryWorkflowDispatch({"prostate-heatmap": dispatch}),
            lease_duration=timedelta(milliseconds=30),
        )
        await scheduler.schedule(self._request())
        task = asyncio.create_task(worker.run_once())

        async def cleanup() -> None:
            """Cancel and join the test-owned background task so no worker is left running.

            Returns:
                None: No return value.
            """
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        self.addAsyncCleanup(cleanup)
        await asyncio.wait_for(stopped.wait(), timeout=0.5)
        with self.assertRaises(TimeoutError):
            await task

    async def test_worker_loop_survives_claim_lease_loss(self) -> None:
        """Verify worker loop survives claim lease loss.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        clock = MutableClock(datetime(2026, 9, 17, 8, tzinfo=UTC))
        stopped = asyncio.Event()
        calls = 0

        async def dispatch(
            _invocation: WorkflowInvocation,
            _execution_context: WorkflowExecutionContext,
        ) -> WorkflowResult:
            """Simulate the scenario computation, using events or injected failures to control completion and ownership races.

            Args:
                _invocation: Unused named inputs, resolved configuration, public access
                    state, and absolute workflow deadline; retained to match the
                    interface.
                _execution_context: Unused private attempt identity and provider context
                    kept separate from the invocation; retained to match the interface.

            Returns:
                WorkflowResult: Named inline outputs and/or references to completed
                    artifact files.
            """
            nonlocal calls
            calls += 1
            if calls == 1:
                clock.advance(timedelta(seconds=2))
                try:
                    await asyncio.Event().wait()
                finally:
                    stopped.set()
            return WorkflowResult(outputs={})

        repository = InMemoryExecutionRepository(
            clock=clock,
            processing_policy=RetryPolicy(3, timedelta(0), timedelta(0)),
        )
        scheduler, worker = self._scheduler_and_worker(
            repository,
            StandaloneProvider(),
            InMemoryWorkflowDispatch({"prostate-heatmap": dispatch}),
            clock=clock,
            lease_duration=timedelta(milliseconds=30),
        )
        admitted = await scheduler.schedule(self._request())
        worker_task = self._start_worker(worker)

        await asyncio.wait_for(stopped.wait(), timeout=1)
        await self._wait_for_status(
            scheduler, admitted.execution.id, ExecutionStatus.SUCCEEDED
        )
        self.assertFalse(worker_task.done())
        self.assertEqual(calls, 2)

    async def test_renewal_error_stops_dispatch_and_discards_late_result(self) -> None:
        """Verify renewal error stops dispatch and discards late result.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        clock = MutableClock(datetime(2026, 9, 17, 8, tzinfo=UTC))
        stopped = asyncio.Event()
        calls = 0

        class FailingRenewalRepository(InMemoryExecutionRepository):
            """Provide the failing renewal repository fixture local to this failure or cancellation scenario."""

            failed = False

            async def renew(
                self,
                execution_id: str,
                fencing_token: str,
                lease_duration: timedelta,
            ) -> RenewDirective:
                """Inject a lease-renewal failure or stall to verify work stops when ownership cannot be maintained.

                Args:
                    execution_id: Durable execution identifier, not an external provider
                        job ID.
                    fencing_token: Current claim token; stale tokens cannot mutate
                        execution state.
                    lease_duration: How long a claimed attempt retains ownership without
                        renewal.

                Returns:
                    RenewDirective: Instruction to continue or stop the current attempt.

                Raises:
                    ExecutionRepositoryUnavailableError: Database access or the storage
                        transaction fails.
                """
                if not self.failed:
                    self.failed = True
                    clock.advance(timedelta(seconds=2))
                    raise ExecutionRepositoryUnavailableError("renewal unavailable")
                return await super().renew(execution_id, fencing_token, lease_duration)

        async def dispatch(
            _invocation: WorkflowInvocation,
            _execution_context: WorkflowExecutionContext,
        ) -> WorkflowResult:
            """Simulate the scenario computation, using events or injected failures to control completion and ownership races.

            Args:
                _invocation: Unused named inputs, resolved configuration, public access
                    state, and absolute workflow deadline; retained to match the
                    interface.
                _execution_context: Unused private attempt identity and provider context
                    kept separate from the invocation; retained to match the interface.

            Returns:
                WorkflowResult: Named inline outputs and/or references to completed
                    artifact files.
            """
            nonlocal calls
            calls += 1
            if calls == 1:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    stopped.set()
                    return WorkflowResult(outputs={"stale": SharedArtifact("/stale")})
            return WorkflowResult(outputs={})

        repository = FailingRenewalRepository(
            clock=clock,
            processing_policy=RetryPolicy(3, timedelta(0), timedelta(0)),
        )
        scheduler, worker = self._scheduler_and_worker(
            repository,
            StandaloneProvider(),
            InMemoryWorkflowDispatch({"prostate-heatmap": dispatch}),
            clock=clock,
            lease_duration=timedelta(milliseconds=30),
        )
        admitted = await scheduler.schedule(self._request())
        worker_task = self._start_worker(worker)

        await asyncio.wait_for(stopped.wait(), timeout=1)
        await self._wait_for_status(
            scheduler, admitted.execution.id, ExecutionStatus.SUCCEEDED
        )
        self.assertFalse(worker_task.done())
        self.assertEqual(calls, 2)
        self.assertEqual(
            (await scheduler.get(admitted.execution.id)).result,
            WorkflowResult(outputs={}),
        )

    async def test_busy_worker_expires_queued_deadlines_without_dispatching_more(
        self,
    ) -> None:
        """Verify busy worker expires queued deadlines without dispatching more.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        clock = MutableClock(datetime(2026, 9, 17, 8, tzinfo=UTC))
        started = asyncio.Event()
        calls = 0

        async def dispatch(
            _invocation: WorkflowInvocation,
            _execution_context: WorkflowExecutionContext,
        ) -> WorkflowResult:
            """Simulate the scenario computation, using events or injected failures to control completion and ownership races.

            Args:
                _invocation: Unused named inputs, resolved configuration, public access
                    state, and absolute workflow deadline; retained to match the
                    interface.
                _execution_context: Unused private attempt identity and provider context
                    kept separate from the invocation; retained to match the interface.

            Returns:
                WorkflowResult: Named inline outputs and/or references to completed
                    artifact files.

            Raises:
                AssertionError: An expected invariant or fixture interaction does not
                    hold.
            """
            nonlocal calls
            calls += 1
            started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        repository = InMemoryExecutionRepository(clock=clock)
        scheduler, worker = self._scheduler_and_worker(
            repository,
            StandaloneProvider(),
            InMemoryWorkflowDispatch({"prostate-heatmap": dispatch}),
            clock=clock,
        )
        running = await scheduler.schedule(self._request("running"))
        queued = await scheduler.schedule(
            ScheduleRequest(
                workflow_id="prostate-heatmap",
                provider="standalone",
                body={},
                idempotency_key="queued",
                requested_timeout_seconds=1,
            )
        )
        worker_task = self._start_worker(worker)
        await asyncio.wait_for(started.wait(), timeout=1)
        clock.advance(timedelta(seconds=2))

        await self._wait_for_status(
            scheduler, queued.execution.id, ExecutionStatus.FAILED
        )
        self.assertEqual((await scheduler.get(running.execution.id)).status, "running")
        self.assertEqual(calls, 1)
        self.assertFalse(worker_task.done())

    async def test_expired_lease_rejects_settlement_before_recovery(self) -> None:
        """Verify expired lease rejects settlement before recovery.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        clock = MutableClock(datetime(2026, 9, 17, 8, tzinfo=UTC))
        repository = InMemoryExecutionRepository(clock=clock)
        scheduler, _worker = self._scheduler_and_worker(
            repository,
            StandaloneProvider(),
            InMemoryWorkflowDispatch({}),
            clock=clock,
        )
        admitted = await scheduler.schedule(self._request())
        claim = await repository.claim_due("worker-1", timedelta(seconds=1))
        assert claim is not None
        clock.advance(timedelta(seconds=2))

        with self.assertRaises(StaleWorkClaimError):
            await repository.settle(
                claim.execution_id,
                claim.fencing_token,
                ProcessingSucceeded(WorkflowResult(outputs={})),
            )
        self.assertIsNone((await repository.get(admitted.execution.id)).result)

    def _start_worker(self, worker: ExecutionWorker) -> asyncio.Task[None]:
        """Start the worker polling loop and register async cleanup that cancels and joins it.

        Args:
            worker: Manually driven execution worker used by the fixture.

        Returns:
            asyncio.Task[None]: Started worker task, with fixture cleanup registered to
                cancel and join it.
        """
        stop = asyncio.Event()
        task = asyncio.create_task(worker.run(stop, poll_interval=0.005))

        async def cleanup() -> None:
            """Cancel and join the test-owned background task so no worker is left running.

            Returns:
                None: No return value.
            """
            stop.set()
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

        self.addAsyncCleanup(cleanup)
        return task

    async def _wait_for_status(
        self, scheduler: ExecutionScheduler, execution_id: str, status: ExecutionStatus
    ) -> None:
        """Poll the in-memory scheduler until the expected status appears within a one-second test timeout.

        Args:
            scheduler: Durable scheduler boundary or its live HTTP client, according to
                the declared type.
            execution_id: Durable execution identifier, not an external provider job ID.
            status: Expected caller-visible execution status.

        Returns:
            None: No return value.
        """
        async with asyncio.timeout(1):
            while (await scheduler.get(execution_id)).status != status:
                await asyncio.sleep(0.005)

    async def test_deadline_failure_is_reported_until_provider_deadline(self) -> None:
        """Verify deadline failure is reported until provider deadline.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        clock = MutableClock(datetime(2026, 9, 17, 8, tzinfo=UTC))
        repository = InMemoryExecutionRepository(clock=clock)
        execution_deadline = clock() + timedelta(minutes=5)
        report_deadline = clock() + timedelta(hours=1)
        admission = Admission(
            identity=SubmissionIdentity("empaia", "job-1", "fingerprint"),
            workflow_id="prostate-heatmap",
            invocation=WorkflowInvocation(
                inputs={},
                resolved_configuration={},
                input_access=InputAccessContext(kind="empaia", state={}),
                deadline_at=execution_deadline,
            ),
            protected_context={"credential": "secret"},
            deadline_at=execution_deadline,
            terminal_report_deadline_at=report_deadline,
        )
        await repository.admit(admission)
        processing = await repository.claim_due("worker-1", timedelta(hours=1))
        assert processing is not None
        clock.advance(timedelta(minutes=6))

        directive = await repository.renew(
            processing.execution_id,
            processing.fencing_token,
            timedelta(minutes=1),
        )
        report = await repository.claim_due("worker-2", timedelta(minutes=1))

        self.assertEqual(directive, "deadline_exceeded")
        assert report is not None
        self.assertEqual(report.kind, WorkKind.PROVIDER_FAILURE_REPORT)
        assert report.error is not None
        self.assertEqual(report.error.code, "execution_deadline_exceeded")
        failed = await repository.settle(
            report.execution_id,
            report.fencing_token,
            ProviderFinishSucceeded(),
        )
        self.assertEqual(failed.status, "failed")
        self.assertIsNone(failed.work_kind)

    async def test_repository_failure_is_translated_by_the_scheduler(self) -> None:
        """Verify repository failure is translated by the scheduler.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        provider = StandaloneProvider()
        scheduler = ExecutionScheduler(
            repository=UnavailableExecutionRepository(),
            dispatch=InMemoryWorkflowDispatch({}),
            providers={provider.name: provider},
            known_workflows={"prostate-heatmap"},
            default_timeout=timedelta(hours=1),
            maximum_timeout=timedelta(hours=6),
        )

        with self.assertRaises(SchedulingUnavailableError):
            await scheduler.schedule(self._request())

    async def test_provider_finish_retry_uses_stored_result_without_redispatch(
        self,
    ) -> None:
        """Verify provider finish retry uses stored result without redispatch.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        now = datetime(2026, 9, 17, 8, tzinfo=UTC)
        result = WorkflowResult(outputs={})
        dispatches = 0

        def dispatch(
            _invocation: WorkflowInvocation,
            _execution_context: WorkflowExecutionContext,
        ) -> WorkflowResult:
            """Simulate the scenario computation, using events or injected failures to control completion and ownership races.

            Args:
                _invocation: Unused named inputs, resolved configuration, public access
                    state, and absolute workflow deadline; retained to match the
                    interface.
                _execution_context: Unused private attempt identity and provider context
                    kept separate from the invocation; retained to match the interface.

            Returns:
                WorkflowResult: Named inline outputs and/or references to completed
                    artifact files.
            """
            nonlocal dispatches
            dispatches += 1
            return result

        provider = RecordingProvider(
            finish_failures=[
                ProviderFinishFailureError(
                    code="provider_temporarily_unavailable",
                    safe_message="provider is temporarily unavailable",
                    retryable=True,
                )
            ]
        )
        repository = InMemoryExecutionRepository(
            clock=lambda: now,
            finish_policy=RetryPolicy(3, timedelta(0), timedelta(0)),
        )
        scheduler, worker = self._scheduler_and_worker(
            repository,
            provider,
            InMemoryWorkflowDispatch({"prostate-heatmap": dispatch}),
            clock=lambda: now,
        )
        admitted = await scheduler.schedule(self._request())

        await worker.run_once()
        await worker.run_once()
        retrying = await scheduler.get(admitted.execution.id)

        self.assertEqual(retrying.status, "running")
        self.assertIsNone(retrying.result)
        await worker.run_once()
        completed = await scheduler.get(admitted.execution.id)
        self.assertEqual(completed.status, "succeeded")
        self.assertEqual(completed.result, result)
        self.assertEqual(dispatches, 1)
        self.assertTrue(
            all(isinstance(outcome, ProviderSuccess) for outcome in provider.outcomes)
        )

    async def test_failure_and_cancellation_reports_do_not_reopen_terminal_state(
        self,
    ) -> None:
        """Verify failure and cancellation reports do not reopen terminal state.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        provider = RecordingProvider()
        repository = InMemoryExecutionRepository()

        def fail(
            _invocation: WorkflowInvocation,
            _execution_context: WorkflowExecutionContext,
        ) -> WorkflowResult:
            """Inject the configured remote failure for safe classification or terminal reporting checks.

            Args:
                _invocation: Unused named inputs, resolved configuration, public access
                    state, and absolute workflow deadline; retained to match the
                    interface.
                _execution_context: Unused private attempt identity and provider context
                    kept separate from the invocation; retained to match the interface.

            Returns:
                WorkflowResult: Named inline outputs and/or references to completed
                    artifact files.

            Raises:
                RemoteWorkflowFailureError: The remote workflow returns a safe
                    structured failure.
            """
            raise RemoteWorkflowFailureError(
                code="invalid_slide",
                safe_message="slide cannot be processed",
                retryable=False,
            )

        scheduler, worker = self._scheduler_and_worker(
            repository,
            provider,
            InMemoryWorkflowDispatch({"prostate-heatmap": fail}),
        )
        failed = await scheduler.schedule(self._request("failure"))

        await worker.run_once()
        terminal_failure = await scheduler.get(failed.execution.id)
        self.assertEqual(terminal_failure.status, "failed")
        await worker.run_once()
        self.assertEqual((await scheduler.get(failed.execution.id)).status, "failed")
        self.assertIsInstance(provider.outcomes[-1], ProviderFailure)

        cancelled = await scheduler.schedule(self._request("cancellation"))
        await scheduler.cancel(cancelled.execution.id)
        await worker.run_once()
        self.assertEqual(
            (await scheduler.get(cancelled.execution.id)).status,
            "cancelled",
        )
        self.assertIsInstance(provider.outcomes[-1], ProviderCancellation)

    async def test_terminal_reporting_is_bounded_without_delaying_failure(self) -> None:
        """Verify terminal reporting is bounded without delaying failure.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        failures = [
            ProviderFinishFailureError(
                code="provider_unavailable",
                safe_message="provider is unavailable",
                retryable=True,
            )
            for _ in range(2)
        ]
        provider = RecordingProvider(finish_failures=failures)
        repository = InMemoryExecutionRepository(
            finish_policy=RetryPolicy(2, timedelta(0), timedelta(0))
        )

        def fail(
            _invocation: WorkflowInvocation,
            _execution_context: WorkflowExecutionContext,
        ) -> WorkflowResult:
            """Inject the configured remote failure for safe classification or terminal reporting checks.

            Args:
                _invocation: Unused named inputs, resolved configuration, public access
                    state, and absolute workflow deadline; retained to match the
                    interface.
                _execution_context: Unused private attempt identity and provider context
                    kept separate from the invocation; retained to match the interface.

            Returns:
                WorkflowResult: Named inline outputs and/or references to completed
                    artifact files.

            Raises:
                RemoteWorkflowFailureError: The remote workflow returns a safe
                    structured failure.
            """
            raise RemoteWorkflowFailureError(
                code="invalid_slide",
                safe_message="slide cannot be processed",
                retryable=False,
            )

        scheduler, worker = self._scheduler_and_worker(
            repository,
            provider,
            InMemoryWorkflowDispatch({"prostate-heatmap": fail}),
        )
        admitted = await scheduler.schedule(self._request())

        await worker.run_once()
        self.assertEqual((await scheduler.get(admitted.execution.id)).status, "failed")
        await worker.run_once()
        await worker.run_once()

        durable = await repository.get(admitted.execution.id)
        self.assertEqual(durable.status, "failed")
        self.assertEqual(durable.finish_attempts, 2)
        self.assertIsNone(durable.work_kind)
        self.assertFalse(await worker.run_once())

    async def test_unknown_provider_finish_outcome_never_retries_publication(
        self,
    ) -> None:
        """Verify unknown provider finish outcome never retries publication.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        provider = RecordingProvider(
            finish_failures=[
                ProviderFinishFailureError(
                    code="provider_outcome_unknown",
                    safe_message="provider publication outcome is unknown",
                    retryable=True,
                    outcome_unknown=True,
                )
            ]
        )
        repository = InMemoryExecutionRepository(
            finish_policy=RetryPolicy(3, timedelta(0), timedelta(0))
        )
        scheduler, worker = self._scheduler_and_worker(
            repository,
            provider,
            InMemoryWorkflowDispatch(
                {
                    "prostate-heatmap": lambda _invocation, _execution_context: (
                        WorkflowResult(outputs={})
                    )
                }
            ),
        )
        admitted = await scheduler.schedule(self._request())

        await worker.run_once()
        await worker.run_once()

        failed = await repository.get(admitted.execution.id)
        self.assertEqual(failed.status, "failed")
        self.assertEqual(failed.finish_attempts, 1)
        self.assertIsNone(failed.work_kind)
        self.assertFalse(await worker.run_once())

    async def test_protected_context_stays_separate_from_durable_invocation(
        self,
    ) -> None:
        """Verify protected context stays separate from durable invocation.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        provider = RecordingProvider(protected_context={"credential": "secret"})
        repository = InMemoryExecutionRepository()
        seen_contexts: list[WorkflowExecutionContext] = []

        def dispatch(
            invocation: WorkflowInvocation, execution_context: WorkflowExecutionContext
        ) -> WorkflowResult:
            """Simulate the scenario computation, using events or injected failures to control completion and ownership races.

            Args:
                invocation: Named inputs, resolved configuration, public access state,
                    and absolute workflow deadline.
                execution_context: Private attempt identity and provider context kept
                    separate from the invocation.

            Returns:
                WorkflowResult: Named inline outputs and/or references to completed
                    artifact files.
            """
            self.assertNotIn("credential", invocation.input_access.state)
            self.assertNotIn("processing_attempt_id", invocation.input_access.state)
            seen_contexts.append(execution_context)
            return WorkflowResult(outputs={})

        scheduler, worker = self._scheduler_and_worker(
            repository,
            provider,
            InMemoryWorkflowDispatch({"prostate-heatmap": dispatch}),
        )
        admitted = await scheduler.schedule(self._request())

        durable = await repository.get(admitted.execution.id)
        self.assertNotIn("credential", durable.invocation.input_access.state)
        await worker.run_once()
        self.assertEqual(seen_contexts[0].protected_context, {"credential": "secret"})
        self.assertIsNotNone(seen_contexts[0].processing_attempt_id)
        durable = await repository.get(admitted.execution.id)
        self.assertNotIn("credential", durable.invocation.input_access.state)
        self.assertNotIn("processing_attempt_id", durable.invocation.input_access.state)

    async def test_scheduled_result_is_visible_after_provider_finish(self) -> None:
        """Verify scheduled result is visible after provider finish.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        now = datetime(2026, 9, 17, 8, tzinfo=UTC)
        result = WorkflowResult(
            outputs={
                "heatmap": SharedArtifact(
                    path="/shared/heatmap.tiff",
                    media_type="image/tiff",
                )
            }
        )
        repository = InMemoryExecutionRepository(clock=lambda: now)
        provider = StandaloneProvider()
        scheduler = ExecutionScheduler(
            repository=repository,
            dispatch=InMemoryWorkflowDispatch({}),
            providers={provider.name: provider},
            known_workflows={"prostate-heatmap"},
            default_timeout=timedelta(hours=1),
            maximum_timeout=timedelta(hours=6),
            clock=lambda: now,
        )
        worker = ExecutionWorker(
            repository=repository,
            active_work=ActiveWorkRegistry(),
            providers={provider.name: provider},
            dispatch=InMemoryWorkflowDispatch(
                {"prostate-heatmap": lambda _invocation, _execution_context: result}
            ),
            owner="worker-1",
            lease_duration=timedelta(minutes=1),
        )

        admitted = await scheduler.schedule(
            ScheduleRequest(
                workflow_id="prostate-heatmap",
                provider="standalone",
                body={},
                idempotency_key="request-1",
            )
        )

        self.assertTrue(admitted.created)
        self.assertEqual(admitted.execution.status, "queued")
        self.assertIsNone(admitted.execution.result)

        self.assertTrue(await worker.run_once())
        processing_complete = await scheduler.get(admitted.execution.id)
        self.assertEqual(processing_complete.status, "running")
        self.assertIsNone(processing_complete.result)

        self.assertTrue(await worker.run_once())
        completed = await scheduler.get(admitted.execution.id)
        self.assertEqual(completed.status, "succeeded")
        self.assertEqual(completed.result, result)

    async def test_replacement_worker_recovers_abandoned_processing(self) -> None:
        """Verify replacement worker recovers abandoned processing.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        clock = MutableClock(datetime(2026, 9, 17, 8, tzinfo=UTC))
        provider = StandaloneProvider()
        repository = InMemoryExecutionRepository(
            clock=clock,
            processing_policy=RetryPolicy(2, timedelta(0), timedelta(0)),
        )
        scheduler = ExecutionScheduler(
            repository=repository,
            dispatch=InMemoryWorkflowDispatch({}),
            providers={provider.name: provider},
            known_workflows={"prostate-heatmap"},
            default_timeout=timedelta(hours=1),
            maximum_timeout=timedelta(hours=6),
            clock=clock,
        )
        abandoned_started = asyncio.Event()

        async def abandoned(
            _invocation: WorkflowInvocation,
            _execution_context: WorkflowExecutionContext,
        ) -> WorkflowResult:
            """Signal that dispatch started and block until its owning worker is interrupted.

            Args:
                _invocation: Unused named inputs, resolved configuration, public access
                    state, and absolute workflow deadline; retained to match the
                    interface.
                _execution_context: Unused private attempt identity and provider context
                    kept separate from the invocation; retained to match the interface.

            Returns:
                WorkflowResult: Named inline outputs and/or references to completed
                    artifact files.

            Raises:
                AssertionError: An expected invariant or fixture interaction does not
                    hold.
            """
            abandoned_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        first_worker = ExecutionWorker(
            repository=repository,
            active_work=ActiveWorkRegistry(),
            providers={provider.name: provider},
            dispatch=InMemoryWorkflowDispatch({"prostate-heatmap": abandoned}),
            owner="scheduler-before-restart",
            lease_duration=timedelta(seconds=30),
        )
        replacement_worker = ExecutionWorker(
            repository=repository,
            active_work=ActiveWorkRegistry(),
            providers={provider.name: provider},
            dispatch=InMemoryWorkflowDispatch(
                {
                    "prostate-heatmap": lambda _invocation, _execution_context: (
                        WorkflowResult(outputs={})
                    )
                }
            ),
            owner="scheduler-after-restart",
            lease_duration=timedelta(seconds=30),
        )
        admitted = await scheduler.schedule(self._request("restart"))
        abandoned_task = asyncio.create_task(first_worker.run_once())
        await abandoned_started.wait()

        abandoned_task.cancel()
        await asyncio.gather(abandoned_task, return_exceptions=True)
        clock.advance(timedelta(seconds=31))

        self.assertTrue(await replacement_worker.run_once())
        recovered = await repository.get(admitted.execution.id)
        self.assertEqual(recovered.processing_attempts, 2)
        self.assertEqual(recovered.status, "running")
        self.assertTrue(await replacement_worker.run_once())
        self.assertEqual(
            (await scheduler.get(admitted.execution.id)).status,
            "succeeded",
        )

    async def test_confirmed_rejection_waits_without_consuming_an_attempt(
        self,
    ) -> None:
        """Verify confirmed rejection waits without consuming an attempt.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        clock = MutableClock(datetime(2026, 9, 17, 8, tzinfo=UTC))
        result = WorkflowResult(outputs={})
        outcomes: list[WorkflowResult | Exception] = [
            ConfirmedDispatchRejectionError(),
            result,
        ]

        def dispatch(
            _invocation: WorkflowInvocation,
            _execution_context: WorkflowExecutionContext,
        ) -> WorkflowResult:
            """Simulate the scenario computation, using events or injected failures to control completion and ownership races.

            Args:
                _invocation: Unused named inputs, resolved configuration, public access
                    state, and absolute workflow deadline; retained to match the
                    interface.
                _execution_context: Unused private attempt identity and provider context
                    kept separate from the invocation; retained to match the interface.

            Returns:
                WorkflowResult: Named inline outputs and/or references to completed
                    artifact files.
            """
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        repository = InMemoryExecutionRepository(
            clock=clock,
            processing_policy=RetryPolicy(
                max_attempts=2,
                initial_backoff=timedelta(seconds=5),
                maximum_backoff=timedelta(seconds=30),
            ),
        )
        provider = StandaloneProvider()
        scheduler = ExecutionScheduler(
            repository=repository,
            dispatch=InMemoryWorkflowDispatch({}),
            providers={provider.name: provider},
            known_workflows={"prostate-heatmap"},
            default_timeout=timedelta(hours=1),
            maximum_timeout=timedelta(hours=6),
            clock=clock,
        )
        worker = ExecutionWorker(
            repository=repository,
            active_work=ActiveWorkRegistry(),
            providers={provider.name: provider},
            dispatch=InMemoryWorkflowDispatch({"prostate-heatmap": dispatch}),
            owner="worker-1",
            lease_duration=timedelta(minutes=1),
        )
        admitted = await scheduler.schedule(
            ScheduleRequest(
                workflow_id="prostate-heatmap",
                provider="standalone",
                body={},
                idempotency_key="request-1",
            )
        )

        self.assertTrue(await worker.run_once())
        waiting = await repository.get(admitted.execution.id)
        self.assertEqual(waiting.status, "queued")
        self.assertEqual(waiting.processing_attempts, 0)
        self.assertEqual(waiting.processing_rejections, 1)
        self.assertFalse(await worker.run_once())

        clock.advance(timedelta(seconds=5))
        self.assertTrue(await worker.run_once())
        accepted = await repository.get(admitted.execution.id)
        self.assertEqual(accepted.processing_attempts, 1)
        self.assertEqual(accepted.status, "running")

    async def test_unknown_dispatch_outcome_consumes_attempts_and_exhausts_policy(
        self,
    ) -> None:
        """Verify unknown dispatch outcome consumes attempts and exhausts policy.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        clock = MutableClock(datetime(2026, 9, 17, 8, tzinfo=UTC))

        def dispatch(
            _invocation: WorkflowInvocation,
            _execution_context: WorkflowExecutionContext,
        ) -> WorkflowResult:
            """Simulate the scenario computation, using events or injected failures to control completion and ownership races.

            Args:
                _invocation: Unused named inputs, resolved configuration, public access
                    state, and absolute workflow deadline; retained to match the
                    interface.
                _execution_context: Unused private attempt identity and provider context
                    kept separate from the invocation; retained to match the interface.

            Returns:
                WorkflowResult: Named inline outputs and/or references to completed
                    artifact files.

            Raises:
                UnknownDispatchOutcomeError: Transport or response decoding cannot
                    establish the remote outcome.
            """
            raise UnknownDispatchOutcomeError

        repository = InMemoryExecutionRepository(
            clock=clock,
            processing_policy=RetryPolicy(
                max_attempts=2,
                initial_backoff=timedelta(0),
                maximum_backoff=timedelta(0),
            ),
        )
        provider = StandaloneProvider()
        scheduler = ExecutionScheduler(
            repository=repository,
            dispatch=InMemoryWorkflowDispatch({}),
            providers={provider.name: provider},
            known_workflows={"prostate-heatmap"},
            default_timeout=timedelta(hours=1),
            maximum_timeout=timedelta(hours=6),
            clock=clock,
        )
        worker = ExecutionWorker(
            repository=repository,
            active_work=ActiveWorkRegistry(),
            providers={provider.name: provider},
            dispatch=InMemoryWorkflowDispatch({"prostate-heatmap": dispatch}),
            owner="worker-1",
            lease_duration=timedelta(minutes=1),
        )
        admitted = await scheduler.schedule(
            ScheduleRequest(
                workflow_id="prostate-heatmap",
                provider="standalone",
                body={},
                idempotency_key="request-1",
            )
        )

        self.assertTrue(await worker.run_once())
        retrying = await repository.get(admitted.execution.id)
        self.assertEqual(retrying.status, "queued")
        self.assertEqual(retrying.processing_attempts, 1)

        self.assertTrue(await worker.run_once())
        failed = await scheduler.get(admitted.execution.id)
        self.assertEqual(failed.status, "failed")
        assert failed.error is not None
        self.assertEqual(failed.error.code, "workflow_outcome_unknown")

    async def test_cancel_fences_then_stops_locally_owned_dispatch(self) -> None:
        """Verify cancel fences then stops locally owned dispatch.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        clock = MutableClock(datetime(2026, 9, 17, 8, tzinfo=UTC))
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def dispatch(
            _invocation: WorkflowInvocation,
            _execution_context: WorkflowExecutionContext,
        ) -> WorkflowResult:
            """Simulate the scenario computation, using events or injected failures to control completion and ownership races.

            Args:
                _invocation: Unused named inputs, resolved configuration, public access
                    state, and absolute workflow deadline; retained to match the
                    interface.
                _execution_context: Unused private attempt identity and provider context
                    kept separate from the invocation; retained to match the interface.

            Returns:
                WorkflowResult: Named inline outputs and/or references to completed
                    artifact files.

            Raises:
                AssertionError: An expected invariant or fixture interaction does not
                    hold.
            """
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
            raise AssertionError("unreachable")

        repository = InMemoryExecutionRepository(clock=clock)
        provider = StandaloneProvider()
        scheduler = ExecutionScheduler(
            repository=repository,
            dispatch=InMemoryWorkflowDispatch({"prostate-heatmap": dispatch}),
            providers={provider.name: provider},
            known_workflows={"prostate-heatmap"},
            default_timeout=timedelta(hours=1),
            maximum_timeout=timedelta(hours=6),
            clock=clock,
            lease_duration=timedelta(minutes=1),
            poll_interval=0.01,
        )
        admitted = await scheduler.schedule(
            ScheduleRequest(
                workflow_id="prostate-heatmap",
                provider="standalone",
                body={},
                idempotency_key="request-1",
            )
        )
        async with scheduler.running():
            await asyncio.wait_for(started.wait(), timeout=1)
            cancelled = await scheduler.cancel(admitted.execution.id)
            repeated = await scheduler.cancel(admitted.execution.id)

            self.assertEqual(cancelled.status, "cancelled")
            self.assertEqual(repeated.status, "cancelled")
            await asyncio.wait_for(stopped.wait(), timeout=1)
        durable = await repository.get(admitted.execution.id)
        self.assertEqual(durable.status, "cancelled")
        self.assertIsNone(durable.fencing_token)

    async def test_live_worker_renews_lease_before_another_worker_can_claim(
        self,
    ) -> None:
        """Verify live worker renews lease before another worker can claim.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        started = asyncio.Event()
        release = asyncio.Event()
        result = WorkflowResult(outputs={})

        async def dispatch(
            _invocation: WorkflowInvocation,
            _execution_context: WorkflowExecutionContext,
        ) -> WorkflowResult:
            """Simulate the scenario computation, using events or injected failures to control completion and ownership races.

            Args:
                _invocation: Unused named inputs, resolved configuration, public access
                    state, and absolute workflow deadline; retained to match the
                    interface.
                _execution_context: Unused private attempt identity and provider context
                    kept separate from the invocation; retained to match the interface.

            Returns:
                WorkflowResult: Named inline outputs and/or references to completed
                    artifact files.
            """
            started.set()
            await release.wait()
            return result

        repository = InMemoryExecutionRepository()
        provider = StandaloneProvider()
        scheduler = ExecutionScheduler(
            repository=repository,
            dispatch=InMemoryWorkflowDispatch({}),
            providers={provider.name: provider},
            known_workflows={"prostate-heatmap"},
            default_timeout=timedelta(hours=1),
            maximum_timeout=timedelta(hours=6),
        )
        dispatch_adapter = InMemoryWorkflowDispatch({"prostate-heatmap": dispatch})
        first_worker = ExecutionWorker(
            repository=repository,
            active_work=ActiveWorkRegistry(),
            providers={provider.name: provider},
            dispatch=dispatch_adapter,
            owner="worker-1",
            lease_duration=timedelta(milliseconds=60),
        )
        replacement_worker = ExecutionWorker(
            repository=repository,
            active_work=ActiveWorkRegistry(),
            providers={provider.name: provider},
            dispatch=dispatch_adapter,
            owner="worker-2",
            lease_duration=timedelta(milliseconds=60),
        )
        await scheduler.schedule(
            ScheduleRequest(
                workflow_id="prostate-heatmap",
                provider="standalone",
                body={},
                idempotency_key="request-1",
            )
        )
        first_task = asyncio.create_task(first_worker.run_once())
        await started.wait()

        await asyncio.sleep(0.12)
        self.assertFalse(await replacement_worker.run_once())
        live = await repository.find_submission(
            SubmissionIdentity("standalone", "request-1", "same-request")
        )
        assert live is not None
        self.assertEqual(live.lease_owner, "worker-1")

        release.set()
        self.assertTrue(await first_task)
        processing_complete = await repository.get(live.id)
        self.assertEqual(processing_complete.status, "running")
        self.assertEqual(processing_complete.result, result)

    async def test_standalone_replay_skips_preparation_and_conflict_is_rejected(
        self,
    ) -> None:
        """Verify standalone replay skips preparation and conflict is rejected.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "slide.svs").touch()
            (root / "roi.tiff").touch()
            clock = MutableClock(datetime(2026, 9, 17, 8, tzinfo=UTC))
            repository = InMemoryExecutionRepository(clock=clock)
            provider = StandaloneProviderAdapter(WorkflowCatalog.default())
            scheduler = ExecutionScheduler(
                repository=repository,
                dispatch=InMemoryWorkflowDispatch({}),
                providers={provider.name: provider},
                known_workflows={"prostate-heatmap"},
                default_timeout=timedelta(hours=1),
                maximum_timeout=timedelta(hours=6),
                clock=clock,
            )
            request = self._standalone_request(root)

            first = await scheduler.schedule(request)
            (root / "slide.svs").unlink()
            replay = await scheduler.schedule(request)

            self.assertTrue(first.created)
            self.assertFalse(replay.created)
            self.assertEqual(replay.execution.id, first.execution.id)

            conflicting_body = dict(request.body)
            conflicting_body["output_path"] = "outputs/other.tiff"
            with self.assertRaises(ScheduleConflictError):
                await scheduler.schedule(
                    ScheduleRequest(
                        workflow_id=request.workflow_id,
                        provider=request.provider,
                        body=conflicting_body,
                        idempotency_key=request.idempotency_key,
                    )
                )

    async def test_invalid_standalone_path_creates_no_execution(self) -> None:
        """Verify invalid standalone path creates no execution.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "roi.tiff").touch()
            clock = MutableClock(datetime(2026, 9, 17, 8, tzinfo=UTC))
            repository = InMemoryExecutionRepository(clock=clock)
            provider = StandaloneProviderAdapter(WorkflowCatalog.default())
            scheduler = ExecutionScheduler(
                repository=repository,
                dispatch=InMemoryWorkflowDispatch({}),
                providers={provider.name: provider},
                known_workflows={"prostate-heatmap"},
                default_timeout=timedelta(hours=1),
                maximum_timeout=timedelta(hours=6),
                clock=clock,
            )
            request = self._standalone_request(root)

            with self.assertRaises(InvalidScheduleRequestError):
                await scheduler.schedule(request)

            (root / "slide.svs").touch()
            admitted = await scheduler.schedule(request)
            self.assertTrue(admitted.created)

    async def test_invalid_standalone_configuration_creates_no_execution(self) -> None:
        """Verify invalid standalone configuration creates no execution.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "slide.svs").touch()
            (root / "roi.tiff").touch()
            clock = MutableClock(datetime(2026, 9, 17, 8, tzinfo=UTC))
            repository = InMemoryExecutionRepository(clock=clock)
            provider = StandaloneProviderAdapter(WorkflowCatalog.default())
            scheduler = ExecutionScheduler(
                repository=repository,
                dispatch=InMemoryWorkflowDispatch({}),
                providers={provider.name: provider},
                known_workflows={"prostate-heatmap"},
                default_timeout=timedelta(hours=1),
                maximum_timeout=timedelta(hours=6),
                clock=clock,
            )
            valid_request = self._standalone_request(root)
            invalid_body = dict(valid_request.body)
            invalid_body["resolved_configuration"] = {"output_path": "heatmap.tiff"}
            invalid_request = ScheduleRequest(
                workflow_id=valid_request.workflow_id,
                provider=valid_request.provider,
                body=invalid_body,
                idempotency_key=valid_request.idempotency_key,
            )

            with self.assertRaises(InvalidScheduleRequestError):
                await scheduler.schedule(invalid_request)

            admitted = await scheduler.schedule(valid_request)
            self.assertTrue(admitted.created)

    @staticmethod
    def _request(key: str = "request-1") -> ScheduleRequest:
        """Build a provider request for the selected fixture identity and deadline settings.

        Args:
            key: Caller replay key identifying this fixture submission.

        Returns:
            ScheduleRequest: Provider-specific admission payload, replay identity, and
                optional timeout.
        """
        return ScheduleRequest(
            workflow_id="prostate-heatmap",
            provider="standalone",
            body={},
            idempotency_key=key,
        )

    @staticmethod
    def _scheduler_and_worker(
        repository: InMemoryExecutionRepository,
        provider: StandaloneProvider,
        dispatch: InMemoryWorkflowDispatch,
        *,
        clock: Callable[[], datetime] | None = None,
        lease_duration: timedelta = timedelta(minutes=1),
    ) -> tuple[ExecutionScheduler, ExecutionWorker]:
        """Construct scheduler and manually driven worker fixtures over the supplied adapters, policies, and clock.

        Args:
            repository: Atomic execution storage boundary; the adapter controls
                durability.
            provider: Provider fixture supplying identity, preparation, and terminal
                outcomes.
            dispatch: Workflow invocation boundary, or a selectable fixture handler in
                lifetime tests.
            clock: Aware timestamp source for admission or deterministic tests; defaults
                to UTC wall time.
            lease_duration: How long a claimed attempt retains ownership without
                renewal.

        Returns:
            tuple[ExecutionScheduler, ExecutionWorker]: Selected records, identifiers,
                or fixture values in the ordering described above.
        """
        scheduler = ExecutionScheduler(
            repository=repository,
            dispatch=InMemoryWorkflowDispatch({}),
            providers={provider.name: provider},
            known_workflows={"prostate-heatmap"},
            default_timeout=timedelta(hours=1),
            maximum_timeout=timedelta(hours=6),
            clock=clock,
        )
        worker = ExecutionWorker(
            repository=repository,
            active_work=ActiveWorkRegistry(),
            providers={provider.name: provider},
            dispatch=dispatch,
            owner="worker-1",
            lease_duration=lease_duration,
        )
        return scheduler, worker

    @staticmethod
    def _standalone_request(root: Path) -> ScheduleRequest:
        """Build a complete shared-storage prostate schedule request rooted at the fixture directory.

        Args:
            root: Existing local directory containing permitted input/output storage.

        Returns:
            ScheduleRequest: Provider-specific admission payload, replay identity, and
                optional timeout.
        """
        return ScheduleRequest(
            workflow_id="prostate-heatmap",
            provider="standalone",
            body={
                "inputs": {
                    "slide": {"kind": "shared_path", "path": "slide.svs"},
                    "region_of_interest": {
                        "kind": "shared_path",
                        "path": "roi.tiff",
                    },
                },
                "output_path": "outputs/heatmap.tiff",
                "input_access": {
                    "kind": "shared_storage",
                    "state": {"root": str(root)},
                },
            },
            idempotency_key="request-1",
        )


if __name__ == "__main__":
    unittest.main()
