from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from psycopg import Error as PsycopgError

from execution_scheduler.contracts import (
    ExecutionError,
    ExecutionStatus,
    ExecutionView,
    InvalidScheduleRequestError,
    ScheduleRequest,
    ScheduleResult,
    SchedulingUnavailableError,
    UnknownWorkflowError,
)
from execution_scheduler.dispatch import (
    ConfirmedDispatchRejectionError,
    RemoteWorkflowFailureError,
    UnknownDispatchOutcomeError,
)
from execution_scheduler.domain import (
    Admission,
    ExecutionRecord,
    ProcessingFailed,
    ProcessingRejected,
    ProcessingSucceeded,
    ProviderFinishFailed,
    ProviderFinishSucceeded,
    RenewDirective,
    StaleWorkClaimError,
    WorkClaim,
    WorkKind,
)
from execution_scheduler.providers import (
    ProviderCancellation,
    ProviderFailure,
    ProviderFinishFailureError,
    ProviderFinishOutcome,
    ProviderSuccess,
)


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from execution_scheduler.dispatch import WorkflowDispatch
    from execution_scheduler.domain import Settlement
    from execution_scheduler.providers import ProviderAdapter
    from execution_scheduler.repository import ExecutionRepository
    from workflow_execution.contracts import WorkflowInvocation


class ActiveWorkRegistry:
    def __init__(self) -> None:
        self._tasks: dict[str, tuple[str, asyncio.Task[Settlement]]] = {}

    def register(
        self,
        execution_id: str,
        fencing_token: str,
        task: asyncio.Task[Settlement],
    ) -> None:
        self._tasks[execution_id] = (fencing_token, task)

    def unregister(self, execution_id: str, fencing_token: str) -> None:
        current = self._tasks.get(execution_id)
        if current is not None and current[0] == fencing_token:
            self._tasks.pop(execution_id)

    def cancel(self, execution_id: str) -> None:
        current = self._tasks.get(execution_id)
        if current is not None:
            current[1].cancel()


class ExecutionScheduler:
    def __init__(
        self,
        *,
        repository: ExecutionRepository,
        providers: Mapping[str, ProviderAdapter],
        known_workflows: set[str] | frozenset[str],
        default_timeout: timedelta,
        maximum_timeout: timedelta,
        clock: Callable[[], datetime] | None = None,
        active_work: ActiveWorkRegistry | None = None,
    ) -> None:
        self._repository = repository
        self._providers = dict(providers)
        self._known_workflows = set(known_workflows)
        self._default_timeout = default_timeout
        self._maximum_timeout = maximum_timeout
        self._clock = clock or (lambda: datetime.now(UTC))
        self._active_work = active_work or ActiveWorkRegistry()

    async def schedule(self, request: ScheduleRequest) -> ScheduleResult:
        if request.workflow_id not in self._known_workflows:
            raise UnknownWorkflowError(request.workflow_id)
        try:
            provider = self._providers[request.provider]
        except KeyError as error:
            raise InvalidScheduleRequestError("unknown provider") from error
        identity = provider.identify(request)
        try:
            existing = await self._repository.find_submission(identity)
        except PsycopgError as error:
            raise SchedulingUnavailableError(
                "execution scheduling is temporarily unavailable"
            ) from error
        if existing is not None:
            return ScheduleResult(self._view(existing), created=False)
        deadline_at = self._clock() + self._requested_timeout(request)
        prepared = await provider.prepare(request, deadline_at)
        if prepared.deadline_cap is not None and prepared.deadline_cap < deadline_at:
            deadline_at = prepared.deadline_cap
        if deadline_at <= self._clock():
            raise InvalidScheduleRequestError(
                "effective execution deadline has expired"
            )
        invocation = replace(prepared.invocation, deadline_at=deadline_at)
        try:
            admitted = await self._repository.admit(
                Admission(
                    identity=identity,
                    workflow_id=request.workflow_id,
                    invocation=invocation,
                    protected_context=prepared.protected_context,
                    deadline_at=deadline_at,
                )
            )
        except PsycopgError as error:
            raise SchedulingUnavailableError(
                "execution scheduling is temporarily unavailable"
            ) from error
        return ScheduleResult(
            self._view(admitted.execution),
            created=admitted.created,
        )

    async def get(self, execution_id: str) -> ExecutionView:
        return self._view(await self._repository.get(execution_id))

    async def cancel(self, execution_id: str) -> ExecutionView:
        execution = await self._repository.request_cancel(execution_id)
        self._active_work.cancel(execution_id)
        return self._view(execution)

    def _requested_timeout(self, request: ScheduleRequest) -> timedelta:
        if request.requested_timeout_seconds is None:
            return self._default_timeout
        if request.requested_timeout_seconds <= 0:
            raise InvalidScheduleRequestError("requested timeout must be positive")
        return min(
            timedelta(seconds=request.requested_timeout_seconds),
            self._maximum_timeout,
        )

    @staticmethod
    def _view(execution: ExecutionRecord) -> ExecutionView:
        return ExecutionView(
            id=execution.id,
            workflow_id=execution.workflow_id,
            status=execution.status,
            result=(
                execution.result
                if execution.status is ExecutionStatus.SUCCEEDED
                else None
            ),
            error=(
                execution.error if execution.status is ExecutionStatus.FAILED else None
            ),
        )


class ExecutionWorker:
    def __init__(
        self,
        *,
        repository: ExecutionRepository,
        providers: Mapping[str, ProviderAdapter],
        dispatch: WorkflowDispatch,
        owner: str,
        lease_duration: timedelta,
        active_work: ActiveWorkRegistry | None = None,
    ) -> None:
        self._repository = repository
        self._providers = dict(providers)
        self._dispatch = dispatch
        self._owner = owner
        self._lease_duration = lease_duration
        self._active_work = active_work or ActiveWorkRegistry()

    async def run(self, stop: asyncio.Event, *, poll_interval: float) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        while not stop.is_set():
            try:
                worked = await self.run_once()
            except Exception as error:  # noqa: BLE001 - isolate worker passes
                logger.error(
                    "execution worker pass failed: %s",
                    type(error).__name__,
                )
                worked = False
            if worked:
                continue
            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=poll_interval)

    async def run_once(self) -> bool:
        claim = await self._repository.claim_due(
            self._owner,
            self._lease_duration,
        )
        if claim is None:
            return False
        task = asyncio.create_task(self._execute(claim))
        self._active_work.register(
            claim.execution_id,
            claim.fencing_token,
            task,
        )
        renewal_task = asyncio.create_task(self._renew_while_running(claim, task))
        try:
            settlement = await task
        except asyncio.CancelledError:
            execution = await self._repository.get(claim.execution_id)
            if (
                execution.status in {ExecutionStatus.CANCELLED, ExecutionStatus.FAILED}
                or execution.fencing_token != claim.fencing_token
            ):
                return True
            raise
        finally:
            renewal_task.cancel()
            await asyncio.gather(renewal_task, return_exceptions=True)
            self._active_work.unregister(claim.execution_id, claim.fencing_token)
        try:
            await self._repository.settle(
                claim.execution_id,
                claim.fencing_token,
                settlement,
            )
        except StaleWorkClaimError:
            return True
        return True

    async def _renew_while_running(
        self,
        claim: WorkClaim,
        task: asyncio.Task[Settlement],
    ) -> None:
        interval = max(self._lease_duration.total_seconds() / 3, 0.001)
        while not task.done():
            await asyncio.sleep(interval)
            if task.done():
                return
            directive = await self._repository.renew(
                claim.execution_id,
                claim.fencing_token,
                self._lease_duration,
            )
            if directive is not RenewDirective.CONTINUE:
                task.cancel()
                return

    async def _execute(self, claim: WorkClaim) -> Settlement:
        settlement: Settlement
        if claim.kind is WorkKind.PROCESSING:
            try:
                result = await self._dispatch.run(
                    claim.workflow_id,
                    claim.fencing_token,
                    _invocation_with_protected_context(claim),
                )
                settlement = ProcessingSucceeded(result)
            except ConfirmedDispatchRejectionError as error:
                settlement = ProcessingRejected(error.retry_after)
            except UnknownDispatchOutcomeError:
                settlement = ProcessingFailed(
                    error=ExecutionError(
                        code="workflow_outcome_unknown",
                        message="workflow dispatch outcome is unknown",
                    ),
                    retryable=True,
                )
            except RemoteWorkflowFailureError as error:
                settlement = ProcessingFailed(
                    error=ExecutionError(error.code, error.safe_message),
                    retryable=error.retryable,
                )
        else:
            try:
                await self._providers[claim.provider].finish(
                    claim.protected_context,
                    _finish_outcome(claim),
                )
                settlement = ProviderFinishSucceeded()
            except ProviderFinishFailureError as error:
                settlement = ProviderFinishFailed(
                    error=ExecutionError(error.code, error.safe_message),
                    retryable=error.retryable,
                    outcome_unknown=error.outcome_unknown,
                )
        return settlement


def _invocation_with_protected_context(claim: WorkClaim) -> WorkflowInvocation:
    access = replace(
        claim.invocation.input_access,
        state={
            **claim.invocation.input_access.state,
            **claim.protected_context,
            "processing_attempt_id": claim.fencing_token,
        },
    )
    return replace(claim.invocation, input_access=access)


def _finish_outcome(claim: WorkClaim) -> ProviderFinishOutcome:
    if claim.kind is WorkKind.PROVIDER_FINISH:
        if claim.result is None:
            raise RuntimeError("provider finish requires a durable result")
        return ProviderSuccess(claim.result)
    if claim.kind is WorkKind.PROVIDER_FAILURE_REPORT:
        if claim.error is None:
            raise RuntimeError("provider failure report requires a durable error")
        return ProviderFailure(claim.error)
    if claim.kind is WorkKind.PROVIDER_CANCELLATION_REPORT:
        return ProviderCancellation()
    raise RuntimeError("unsupported provider work kind")
