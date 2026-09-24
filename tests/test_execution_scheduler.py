from __future__ import annotations

import asyncio
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

from execution_scheduler.active_work import ActiveWorkRegistry
from execution_scheduler.contracts import (
    InvalidScheduleRequestError,
    ScheduleConflictError,
    ScheduleRequest,
    SchedulingUnavailableError,
)
from execution_scheduler.dispatch import (
    ConfirmedDispatchRejectionError,
    InMemoryWorkflowDispatch,
    RemoteWorkflowFailureError,
    UnknownDispatchOutcomeError,
)
from execution_scheduler.domain import (
    Admission,
    PreparedWorkflow,
    ProviderFinishSucceeded,
    RetryPolicy,
    SubmissionIdentity,
    WorkKind,
)
from execution_scheduler.providers import (
    ProviderCancellation,
    ProviderFailure,
    ProviderFinishFailureError,
    ProviderSuccess,
)
from execution_scheduler.repository import (
    ExecutionRepositoryUnavailableError,
    InMemoryExecutionRepository,
)
from execution_scheduler.scheduler import ExecutionScheduler
from execution_scheduler.standalone import StandaloneProviderAdapter
from execution_scheduler.worker import ExecutionWorker
from workflow_execution.catalog import WorkflowCatalog
from workflow_execution.contracts import (
    InputAccessContext,
    SharedArtifact,
    WorkflowInvocation,
    WorkflowResult,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from execution_scheduler.providers import BeforeMutation, ProviderFinishOutcome
    from workflow_execution.contracts import JsonValue


class StandaloneProvider:
    name = "standalone"

    def identify(self, request: ScheduleRequest) -> SubmissionIdentity:
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
        del before_mutation
        return None


class RecordingProvider(StandaloneProvider):
    def __init__(
        self,
        *,
        protected_context: dict[str, JsonValue] | None = None,
        finish_failures: list[ProviderFinishFailureError] | None = None,
    ) -> None:
        self.protected_context = protected_context or {}
        self.finish_failures = finish_failures or []
        self.outcomes: list[ProviderFinishOutcome] = []

    async def prepare(
        self,
        request: ScheduleRequest,
        deadline_at: datetime,
    ) -> PreparedWorkflow:
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
        del before_mutation
        self.outcomes.append(outcome)
        if self.finish_failures:
            raise self.finish_failures.pop(0)


class MutableClock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, duration: timedelta) -> None:
        self.now += duration


class UnavailableExecutionRepository(InMemoryExecutionRepository):
    async def find_submission(
        self,
        _identity: SubmissionIdentity,
    ) -> None:
        raise ExecutionRepositoryUnavailableError("database unavailable")


class ExecutionSchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_deadline_failure_is_reported_until_provider_deadline(self) -> None:
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
        provider = StandaloneProvider()
        scheduler = ExecutionScheduler(
            repository=UnavailableExecutionRepository(),
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
        now = datetime(2026, 9, 17, 8, tzinfo=UTC)
        result = WorkflowResult(outputs={})
        dispatches = 0

        def dispatch(_invocation: WorkflowInvocation) -> WorkflowResult:
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
        provider = RecordingProvider()
        repository = InMemoryExecutionRepository()

        def fail(_invocation: WorkflowInvocation) -> WorkflowResult:
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

        def fail(_invocation: WorkflowInvocation) -> WorkflowResult:
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
                {"prostate-heatmap": lambda _invocation: WorkflowResult(outputs={})}
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

    async def test_protected_context_is_injected_only_for_dispatch(self) -> None:
        provider = RecordingProvider(protected_context={"credential": "secret"})
        repository = InMemoryExecutionRepository()
        seen_state: dict[str, JsonValue] = {}

        def dispatch(invocation: WorkflowInvocation) -> WorkflowResult:
            seen_state.update(invocation.input_access.state)
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
        self.assertEqual(seen_state["credential"], "secret")

    async def test_scheduled_result_is_visible_after_provider_finish(self) -> None:
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
            providers={provider.name: provider},
            known_workflows={"prostate-heatmap"},
            default_timeout=timedelta(hours=1),
            maximum_timeout=timedelta(hours=6),
            clock=lambda: now,
        )
        worker = ExecutionWorker(
            repository=repository,
            providers={provider.name: provider},
            dispatch=InMemoryWorkflowDispatch(
                {"prostate-heatmap": lambda _invocation: result}
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
        clock = MutableClock(datetime(2026, 9, 17, 8, tzinfo=UTC))
        provider = StandaloneProvider()
        repository = InMemoryExecutionRepository(
            clock=clock,
            processing_policy=RetryPolicy(2, timedelta(0), timedelta(0)),
        )
        scheduler = ExecutionScheduler(
            repository=repository,
            providers={provider.name: provider},
            known_workflows={"prostate-heatmap"},
            default_timeout=timedelta(hours=1),
            maximum_timeout=timedelta(hours=6),
            clock=clock,
        )
        abandoned_started = asyncio.Event()

        async def abandoned(_invocation: WorkflowInvocation) -> WorkflowResult:
            abandoned_started.set()
            await asyncio.Event().wait()
            raise AssertionError("unreachable")

        first_worker = ExecutionWorker(
            repository=repository,
            providers={provider.name: provider},
            dispatch=InMemoryWorkflowDispatch({"prostate-heatmap": abandoned}),
            owner="scheduler-before-restart",
            lease_duration=timedelta(seconds=30),
        )
        replacement_worker = ExecutionWorker(
            repository=repository,
            providers={provider.name: provider},
            dispatch=InMemoryWorkflowDispatch(
                {"prostate-heatmap": lambda _invocation: WorkflowResult(outputs={})}
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
        clock = MutableClock(datetime(2026, 9, 17, 8, tzinfo=UTC))
        result = WorkflowResult(outputs={})
        outcomes: list[WorkflowResult | Exception] = [
            ConfirmedDispatchRejectionError(),
            result,
        ]

        def dispatch(_invocation: WorkflowInvocation) -> WorkflowResult:
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
            providers={provider.name: provider},
            known_workflows={"prostate-heatmap"},
            default_timeout=timedelta(hours=1),
            maximum_timeout=timedelta(hours=6),
            clock=clock,
        )
        worker = ExecutionWorker(
            repository=repository,
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
        clock = MutableClock(datetime(2026, 9, 17, 8, tzinfo=UTC))

        def dispatch(_invocation: WorkflowInvocation) -> WorkflowResult:
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
            providers={provider.name: provider},
            known_workflows={"prostate-heatmap"},
            default_timeout=timedelta(hours=1),
            maximum_timeout=timedelta(hours=6),
            clock=clock,
        )
        worker = ExecutionWorker(
            repository=repository,
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
        clock = MutableClock(datetime(2026, 9, 17, 8, tzinfo=UTC))
        started = asyncio.Event()
        stopped = asyncio.Event()

        async def dispatch(_invocation: WorkflowInvocation) -> WorkflowResult:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()
            raise AssertionError("unreachable")

        repository = InMemoryExecutionRepository(clock=clock)
        provider = StandaloneProvider()
        active_work = ActiveWorkRegistry()
        scheduler = ExecutionScheduler(
            repository=repository,
            providers={provider.name: provider},
            known_workflows={"prostate-heatmap"},
            default_timeout=timedelta(hours=1),
            maximum_timeout=timedelta(hours=6),
            clock=clock,
            active_work=active_work,
        )
        worker = ExecutionWorker(
            repository=repository,
            providers={provider.name: provider},
            dispatch=InMemoryWorkflowDispatch({"prostate-heatmap": dispatch}),
            owner="worker-1",
            lease_duration=timedelta(minutes=1),
            active_work=active_work,
        )
        admitted = await scheduler.schedule(
            ScheduleRequest(
                workflow_id="prostate-heatmap",
                provider="standalone",
                body={},
                idempotency_key="request-1",
            )
        )
        worker_task = asyncio.create_task(worker.run_once())
        await started.wait()

        cancelled = await scheduler.cancel(admitted.execution.id)
        repeated = await scheduler.cancel(admitted.execution.id)

        self.assertEqual(cancelled.status, "cancelled")
        self.assertEqual(repeated.status, "cancelled")
        await asyncio.wait_for(stopped.wait(), timeout=1)
        self.assertTrue(await worker_task)
        durable = await repository.get(admitted.execution.id)
        self.assertEqual(durable.status, "cancelled")
        self.assertIsNone(durable.fencing_token)

    async def test_live_worker_renews_lease_before_another_worker_can_claim(
        self,
    ) -> None:
        started = asyncio.Event()
        release = asyncio.Event()
        result = WorkflowResult(outputs={})

        async def dispatch(_invocation: WorkflowInvocation) -> WorkflowResult:
            started.set()
            await release.wait()
            return result

        repository = InMemoryExecutionRepository()
        provider = StandaloneProvider()
        scheduler = ExecutionScheduler(
            repository=repository,
            providers={provider.name: provider},
            known_workflows={"prostate-heatmap"},
            default_timeout=timedelta(hours=1),
            maximum_timeout=timedelta(hours=6),
        )
        dispatch_adapter = InMemoryWorkflowDispatch({"prostate-heatmap": dispatch})
        first_worker = ExecutionWorker(
            repository=repository,
            providers={provider.name: provider},
            dispatch=dispatch_adapter,
            owner="worker-1",
            lease_duration=timedelta(milliseconds=60),
        )
        replacement_worker = ExecutionWorker(
            repository=repository,
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
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "slide.svs").touch()
            (root / "roi.tiff").touch()
            clock = MutableClock(datetime(2026, 9, 17, 8, tzinfo=UTC))
            repository = InMemoryExecutionRepository(clock=clock)
            provider = StandaloneProviderAdapter(WorkflowCatalog.default())
            scheduler = ExecutionScheduler(
                repository=repository,
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
            conflicting_body["resolved_configuration"] = {
                "output_path": "outputs/other.tiff",
                "stride_fraction": 0.5,
                "output_tile_height": 256,
                "output_tile_width": 256,
            }
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
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "roi.tiff").touch()
            clock = MutableClock(datetime(2026, 9, 17, 8, tzinfo=UTC))
            repository = InMemoryExecutionRepository(clock=clock)
            provider = StandaloneProviderAdapter(WorkflowCatalog.default())
            scheduler = ExecutionScheduler(
                repository=repository,
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
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "slide.svs").touch()
            (root / "roi.tiff").touch()
            clock = MutableClock(datetime(2026, 9, 17, 8, tzinfo=UTC))
            repository = InMemoryExecutionRepository(clock=clock)
            provider = StandaloneProviderAdapter(WorkflowCatalog.default())
            scheduler = ExecutionScheduler(
                repository=repository,
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
    ) -> tuple[ExecutionScheduler, ExecutionWorker]:
        scheduler = ExecutionScheduler(
            repository=repository,
            providers={provider.name: provider},
            known_workflows={"prostate-heatmap"},
            default_timeout=timedelta(hours=1),
            maximum_timeout=timedelta(hours=6),
            clock=clock,
        )
        worker = ExecutionWorker(
            repository=repository,
            providers={provider.name: provider},
            dispatch=dispatch,
            owner="worker-1",
            lease_duration=timedelta(minutes=1),
        )
        return scheduler, worker

    @staticmethod
    def _standalone_request(root: Path) -> ScheduleRequest:
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
                "resolved_configuration": {
                    "output_path": "outputs/heatmap.tiff",
                    "stride_fraction": 0.5,
                    "output_tile_height": 256,
                    "output_tile_width": 256,
                },
                "input_access": {
                    "kind": "shared_storage",
                    "state": {"root": str(root)},
                },
            },
            idempotency_key="request-1",
        )


if __name__ == "__main__":
    unittest.main()
