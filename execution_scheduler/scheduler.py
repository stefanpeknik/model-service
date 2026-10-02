from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

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
from execution_scheduler.worker import ExecutionWorker


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Mapping

    from execution_scheduler.dispatch import WorkflowDispatch
    from execution_scheduler.providers import ProviderAdapter
    from execution_scheduler.repository import ExecutionRepository


class ExecutionScheduler:
    def __init__(
        self,
        *,
        repository: ExecutionRepository,
        dispatch: WorkflowDispatch,
        providers: Mapping[str, ProviderAdapter],
        known_workflows: set[str] | frozenset[str],
        default_timeout: timedelta,
        maximum_timeout: timedelta,
        clock: Callable[[], datetime] | None = None,
        lease_duration: timedelta = timedelta(seconds=60),
        poll_interval: float = 0.5,
    ) -> None:
        if lease_duration <= timedelta(0):
            raise ValueError("lease_duration must be positive")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        self._repository = repository
        self._providers = dict(providers)
        self._known_workflows = set(known_workflows)
        self._default_timeout = default_timeout
        self._maximum_timeout = maximum_timeout
        self._clock = clock or (lambda: datetime.now(UTC))
        self._active_work = ActiveWorkRegistry()
        self._worker = ExecutionWorker(
            repository=repository,
            providers=self._providers,
            dispatch=dispatch,
            owner=f"execution-scheduler-{uuid4()}",
            lease_duration=lease_duration,
            active_work=self._active_work,
        )
        self._poll_interval = poll_interval
        self._running = False

    @asynccontextmanager
    async def running(self) -> AsyncIterator[None]:
        if self._running:
            raise RuntimeError("execution scheduler is already running")
        self._running = True
        stop = asyncio.Event()
        try:
            async with asyncio.TaskGroup() as tasks:
                worker_task = tasks.create_task(
                    self._worker.run(stop, poll_interval=self._poll_interval)
                )
                try:
                    yield
                finally:
                    stop.set()
                    worker_task.cancel()
        finally:
            self._running = False

    async def check_ready(self) -> None:
        try:
            await self._repository.check_ready()
        except ExecutionRepositoryUnavailableError as error:
            raise SchedulingUnavailableError(
                "execution repository is temporarily unavailable"
            ) from error

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
                    terminal_report_deadline_at=(prepared.terminal_report_deadline_at),
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
        try:
            execution = await self._repository.get(execution_id)
        except ExecutionRepositoryUnavailableError as error:
            raise SchedulingUnavailableError(
                "execution repository is temporarily unavailable"
            ) from error
        return self._view(execution)

    async def cancel(self, execution_id: str) -> ExecutionView:
        try:
            execution = await self._repository.request_cancel(execution_id)
        except ExecutionRepositoryUnavailableError as error:
            raise SchedulingUnavailableError(
                "execution repository is temporarily unavailable"
            ) from error
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
