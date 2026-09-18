from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from execution_scheduler.active_work import ActiveWorkRegistry
from execution_scheduler.contracts import (
    ExecutionStatus,
    ExecutionView,
    InvalidScheduleRequestError,
    ScheduleRequest,
    ScheduleResult,
    SchedulingUnavailableError,
    UnknownWorkflowError,
)
from execution_scheduler.domain import (
    Admission,
    ExecutionRecord,
)
from execution_scheduler.repository import ExecutionRepositoryUnavailableError


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from execution_scheduler.providers import ProviderAdapter
    from execution_scheduler.repository import ExecutionRepository


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
        except ExecutionRepositoryUnavailableError as error:
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
        except ExecutionRepositoryUnavailableError as error:
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
