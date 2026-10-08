"""Own durable admission, public status projection, cancellation, and worker lifetime.

This layer selects provider adapters and coordinates repository calls.
ExecutionTransitions decides state changes; ExecutionWorker performs leased work.
"""

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

    from execution_scheduler.providers import ProviderAdapter
    from execution_scheduler.repository import ExecutionRepository
    from workflow_execution.dispatch import WorkflowDispatch


class ExecutionScheduler:
    """Coordinate public operations and one background worker over injected repository and dispatch interfaces."""

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
        """Validate timing settings and wire a worker with a private local cancellation registry. No tasks start here.

        Args:
            repository: Atomic execution storage boundary; the adapter controls
                durability.
            dispatch: Workflow transport used to invoke the selected Ray Serve
                application.
            providers: Provider name to integration adapter mapping enabled in this
                scheduler.
            known_workflows: Declared workflow IDs accepted by scheduler admission.
            default_timeout: End-to-end budget when the schedule request omits a
                timeout.
            maximum_timeout: Upper bound on a requested end-to-end execution budget.
            clock: Aware timestamp source for admission or deterministic tests; defaults
                to UTC wall time.
            lease_duration: How long a claimed attempt retains ownership without
                renewal.
            poll_interval: Positive interval between maintenance passes and empty/failed
                claim polls.

        Raises:
            ValueError: lease_duration or poll_interval is nonpositive.
        """
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
        """Start and join worker tasks for this context's lifetime.

        Concurrent entry is rejected. Exiting stops local tasks without cancelling
        durable executions, which another worker can recover after lease expiry.
        Injected resources remain caller-owned.

        Yields:
            None: Control while application or scheduler resources remain active.

        Raises:
            RuntimeError: This scheduler lifetime is already active.
            asyncio.CancelledError: The owning task is cancelled; worker tasks are
                joined before propagation.
            ExceptionGroup: The context body or a background task fails inside the task
                group.
        """
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
        """Translate repository readiness failures into a public scheduling availability error.

        Returns:
            None: No return value.

        Raises:
            SchedulingUnavailableError: Admission or durable storage is temporarily
                unavailable.
        """
        try:
            await self._repository.check_ready()
        except ExecutionRepositoryUnavailableError as error:
            raise SchedulingUnavailableError(
                "execution repository is temporarily unavailable"
            ) from error

    async def schedule(self, request: ScheduleRequest) -> ScheduleResult:
        """Identify and replay a submission, prepare new inputs, then admit the complete execution.

        Replay lookup precedes provider preparation. The timeout begins before
        preparation and may be capped by the provider; atomic admission resolves races
        after the early lookup.

        Args:
            request: Provider submission with workflow ID, caller body, replay key, and
                optional execution budget.

        Returns:
            ScheduleResult: Public view of the new or replayed execution; created is
                True only after inserting a new record.

        Raises:
            InvalidScheduleRequestError: Provider payload, configuration, paths, or
                effective deadline are invalid.
            ScheduleConflictError: A provider submission key already exists with
                different request contents.
            SchedulingUnavailableError: Admission or durable storage is temporarily
                unavailable.
            UnknownWorkflowError: The workflow is not declared in the configured
                catalog.
        """
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
        """Read a durable record and project only its caller-visible status and terminal outcome.

        Args:
            execution_id: Durable execution identifier, not an external provider job ID.

        Returns:
            ExecutionView: Current public status, hiding stored results until succeeded
                and errors until failed.

        Raises:
            SchedulingUnavailableError: Admission or durable storage is temporarily
                unavailable.
            UnknownExecutionError: The execution ID is absent from storage.
        """
        try:
            execution = await self._repository.get(execution_id)
        except ExecutionRepositoryUnavailableError as error:
            raise SchedulingUnavailableError(
                "execution repository is temporarily unavailable"
            ) from error
        return self._view(execution)

    async def cancel(self, execution_id: str) -> ExecutionView:
        """Record cancellation and fence the current claim before requesting cancellation of any local task.

        Args:
            execution_id: Durable execution identifier, not an external provider job ID.

        Returns:
            ExecutionView: Cancelled execution view, or the unchanged view if an
                existing terminal outcome already won.

        Raises:
            SchedulingUnavailableError: Admission or durable storage is temporarily
                unavailable.
            UnknownExecutionError: The execution ID is absent from storage.
        """
        try:
            execution = await self._repository.request_cancel(execution_id)
        except ExecutionRepositoryUnavailableError as error:
            raise SchedulingUnavailableError(
                "execution repository is temporarily unavailable"
            ) from error
        self._active_work.cancel(execution_id)
        return self._view(execution)

    def _requested_timeout(self, request: ScheduleRequest) -> timedelta:
        """Use the default timeout or clamp a positive requested timeout to the configured maximum.

        Args:
            request: Provider submission with workflow ID, caller body, replay key, and
                optional execution budget.

        Returns:
            timedelta: Default or clamped positive end-to-end execution duration.

        Raises:
            InvalidScheduleRequestError: Provider payload, configuration, paths, or
                effective deadline are invalid.
        """
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
        """Hide internal fields and unfinished results when converting a record to the public execution view.

        Args:
            execution: Current immutable execution record; transitions return a
                replacement without mutating it.

        Returns:
            ExecutionView: Public status and applicable terminal outcome, excluding
                leases, counters, and private provider context.
        """
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
