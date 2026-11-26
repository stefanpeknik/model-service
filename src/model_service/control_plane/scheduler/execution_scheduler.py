"""Own durable admission, public status projection, cancellation, and worker lifetime.

This layer selects provider adapters and coordinates repository calls.
ExecutionTransitions decides state changes; ExecutionWorker performs leased work.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

from model_service.contracts.failures import InvalidInvocation
from model_service.contracts.log_context import EXECUTION_ID
from model_service.control_plane.pipeline_requests import check_invocation
from model_service.control_plane.scheduler.internal_state import (
    Admission,
    ExecutionRecord,
    WorkKind,
)
from model_service.control_plane.scheduler.public import (
    ExecutionView,
    InvalidScheduleRequestError,
    ScheduleResult,
    SchedulingUnavailableError,
)
from model_service.control_plane.scheduler.running_work import ActiveWorkRegistry
from model_service.control_plane.scheduler.status import ExecutionStatus
from model_service.control_plane.scheduler.store.storage import (
    ExecutionRepositoryUnavailableError,
)
from model_service.control_plane.scheduler.targets import ExecutionTarget
from model_service.control_plane.scheduler.worker import ExecutionWorker


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator, Mapping

    from model_service.contracts.invocation import Invocation
    from model_service.control_plane.scheduler.protocols import PipelineDispatch
    from model_service.control_plane.scheduler.public import (
        PipelineResolver,
        PipelineScheduleRequest,
        ProviderName,
        SubmissionIdentity,
    )
    from model_service.control_plane.scheduler.store.repository import (
        ExecutionRepository,
    )
    from model_service.providers.provider import ProviderAdapter


REPOSITORY_UNAVAILABLE = "execution repository is temporarily unavailable"
SCHEDULING_UNAVAILABLE = "execution scheduling is temporarily unavailable"


@contextmanager
def _unavailable_as_scheduling(message: str = REPOSITORY_UNAVAILABLE) -> Iterator[None]:
    """Report an unavailable repository as unavailable scheduling.

    Args:
        message: The safe message callers see.

    Raises:
        SchedulingUnavailableError: If the block's repository call is unavailable.
    """
    try:
        yield
    except ExecutionRepositoryUnavailableError as error:
        raise SchedulingUnavailableError(message) from error


class ExecutionScheduler:
    """Admission, status, cancellation and the background worker's lifetime."""

    def __init__(
        self,
        *,
        repository: ExecutionRepository,
        dispatch: PipelineDispatch,
        providers: Mapping[ProviderName, ProviderAdapter],
        resolve_pipeline: PipelineResolver,
        default_timeout: timedelta,
        maximum_timeout: timedelta,
        clock: Callable[[], datetime] | None = None,
        lease_duration: timedelta = timedelta(seconds=300),
        poll_interval: float = 0.5,
        max_concurrent: int = 1,
    ) -> None:
        """Validate timing settings and wire a worker with a private local cancellation registry.

        No tasks start here; `running` starts the worker.

        Args:
            repository: Durable execution store.
            dispatch: Client that runs pipelines on the runner.
            providers: Provider adapters by provider name.
            resolve_pipeline: Resolver of a deployed pipeline with a run's
                settings.
            default_timeout: Execution timeout when the request gives none.
            maximum_timeout: Upper bound on a requested timeout.
            clock: Source of the current time; defaults to UTC now.
            lease_duration: How long a claim stays valid without renewal.
            poll_interval: Seconds between worker polls for due work.
            max_concurrent: Maximum attempts this process runs at once.

        Raises:
            ValueError: lease_duration or poll_interval is not positive.
        """
        if lease_duration <= timedelta(0):
            raise ValueError("lease_duration must be positive")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        self._repository = repository
        self._providers = dict(providers)
        self._resolve_pipeline = resolve_pipeline
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
            max_concurrent=max_concurrent,
        )
        self._poll_interval = poll_interval
        self._running = False
        self._worker_task: asyncio.Task[None] | None = None

    @asynccontextmanager
    async def running(self) -> AsyncIterator[None]:
        """Start and join worker tasks for this context's lifetime.

        On exit the worker is stopped and cancelled.

        Yields:
            None while the worker runs.

        Raises:
            RuntimeError: The scheduler is already running.
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
                self._worker_task = worker_task
                try:
                    yield
                finally:
                    stop.set()
                    worker_task.cancel()
        finally:
            self._running = False
            self._worker_task = None

    @property
    def alive(self) -> bool:
        """Whether the worker still claims work; a process without one is useless."""
        return self._worker_task is not None and not self._worker_task.done()

    async def check_ready(self) -> None:
        """Translate repository readiness failures into a public scheduling availability error.

        Raises:
            SchedulingUnavailableError: The execution repository is unavailable.
        """
        with _unavailable_as_scheduling():
            await self._repository.check_ready()

    async def schedule(self, request: PipelineScheduleRequest) -> ScheduleResult:
        """Identify and replay a submission, prepare new inputs, then admit the complete execution.

        A replayed submission returns the existing execution without preparing again.
        The deadline is the requested timeout, shortened to the provider's deadline cap.
        A pipeline is resolved with the prepared invocation's settings, and
        the execution stores the resolved document every attempt sends.

        Args:
            request: Provider schedule request.

        Returns:
            The execution view and whether this call created it.

        Raises:
            UnknownPipelineError: No deployed pipeline has the ID.
            InvalidScheduleRequestError: The provider is unknown, the timeout is not positive, the effective deadline has already passed, or the pipeline refuses the settings or inputs.
            ScheduleConflictError: The idempotency key was used for a different request.
            SchedulingUnavailableError: The execution repository is unavailable.
        """
        self._resolve_pipeline(request.pipeline_id, None)
        try:
            provider = self._providers[request.provider]
        except KeyError as error:
            raise InvalidScheduleRequestError("unknown provider") from error
        identity = provider.identify(request)
        if (existing := await self._find(identity)) is not None:
            # The rest of the request's log lines name the execution.
            EXECUTION_ID.set(existing.id)
            return ScheduleResult(self._view(existing), created=False)
        deadline_at = self._clock() + self._requested_timeout(
            request.requested_timeout_seconds
        )
        prepared = await provider.prepare(request, deadline_at)
        if prepared.deadline_cap is not None and prepared.deadline_cap < deadline_at:
            deadline_at = prepared.deadline_cap
        if deadline_at <= self._clock():
            raise InvalidScheduleRequestError(
                "effective execution deadline has expired"
            )
        invocation = replace(prepared.invocation, deadline_at=deadline_at)
        target, invocation = self._resolved(request.pipeline_id, invocation)
        return await self._admit(
            Admission(
                identity=identity,
                target=target,
                invocation=invocation,
                protected_context=prepared.protected_context,
                deadline_at=deadline_at,
                terminal_report_deadline_at=(prepared.terminal_report_deadline_at),
                priority=prepared.priority,
            )
        )

    def _resolved(
        self, pipeline_id: str, invocation: Invocation
    ) -> tuple[ExecutionTarget, Invocation]:
        """Resolve the pipeline with the invocation's settings and check the invocation.

        Args:
            pipeline_id: The pipeline to run.
            invocation: The prepared invocation; its configuration holds the
                caller's settings.

        Returns:
            The target with the resolved document, and the invocation
            recording every setting the run uses.

        Raises:
            UnknownPipelineError: The pipeline is no longer deployed.
            InvalidScheduleRequestError: The pipeline refuses the settings,
                inputs or output path.
        """
        try:
            resolved = self._resolve_pipeline(pipeline_id, invocation.configuration)
            invocation = replace(invocation, configuration=resolved.pipeline.settings())
            check_invocation(resolved.pipeline, invocation)
        except InvalidInvocation as error:
            raise InvalidScheduleRequestError(str(error)) from error
        return ExecutionTarget.of(resolved), invocation

    async def _find(self, identity: SubmissionIdentity) -> ExecutionRecord | None:
        """Look up an earlier execution with the same submission identity.

        Args:
            identity: Replay identity of the submission.

        Returns:
            The existing execution, or None if this is a new submission.

        Raises:
            ScheduleConflictError: The key was used for a request with a different fingerprint.
            SchedulingUnavailableError: The execution repository is unavailable.
        """
        with _unavailable_as_scheduling(SCHEDULING_UNAVAILABLE):
            return await self._repository.find_submission(identity)

    async def _admit(self, admission: Admission) -> ScheduleResult:
        """Store a new execution, or return the one a concurrent submission admitted first.

        Args:
            admission: Complete execution to admit.

        Returns:
            The execution view and whether this call created it.

        Raises:
            ScheduleConflictError: A concurrent submission used the key for a different request.
            SchedulingUnavailableError: The execution repository is unavailable.
        """
        with _unavailable_as_scheduling(SCHEDULING_UNAVAILABLE):
            admitted = await self._repository.admit(admission)
        # The rest of the request's log lines name the execution.
        EXECUTION_ID.set(admitted.execution.id)
        return ScheduleResult(self._view(admitted.execution), created=admitted.created)

    async def get(self, execution_id: str) -> ExecutionView:
        """Read a durable record and project only its caller-visible status and terminal outcome.

        Args:
            execution_id: ID of the execution.

        Returns:
            The public execution view.

        Raises:
            UnknownExecutionError: No execution has this ID.
            SchedulingUnavailableError: The execution repository is unavailable.
        """
        with _unavailable_as_scheduling():
            execution = await self._repository.get(execution_id)
        return self._view(execution)

    async def get_many(self, execution_ids: list[str]) -> list[ExecutionView]:
        """Read several executions, skipping IDs that do not exist.

        Args:
            execution_ids: IDs to read.

        Returns:
            Views of the executions that exist, in the requested order.

        Raises:
            SchedulingUnavailableError: The execution repository is unavailable.
        """
        with _unavailable_as_scheduling():
            executions = await self._repository.get_many(execution_ids)
        return [self._view(execution) for execution in executions]

    async def cancel(self, execution_id: str) -> ExecutionView:
        """Record cancellation and fence the current claim, then cancel any local task.

        Args:
            execution_id: ID of the execution to cancel.

        Returns:
            The execution view after the cancellation request.

        Raises:
            UnknownExecutionError: No execution has this ID.
            SchedulingUnavailableError: The execution repository is unavailable.
        """
        with _unavailable_as_scheduling():
            execution = await self._repository.request_cancel(execution_id)
        self._active_work.cancel(execution_id)
        return self._view(execution)

    def _requested_timeout(self, seconds: int | None) -> timedelta:
        """Use the default timeout or clamp a positive requested timeout to the configured maximum.

        Args:
            seconds: Requested timeout in seconds, or None for the default.

        Returns:
            The timeout to apply.

        Raises:
            InvalidScheduleRequestError: The requested timeout is not positive.
        """
        if seconds is None:
            return self._default_timeout
        if seconds <= 0:
            raise InvalidScheduleRequestError("requested timeout must be positive")
        return min(timedelta(seconds=seconds), self._maximum_timeout)

    @staticmethod
    def _view(execution: ExecutionRecord) -> ExecutionView:
        """Convert a record to the public execution view.

        Hides internal fields; the result appears only on success and the error only on failure.

        Args:
            execution: Durable execution record.

        Returns:
            The caller-visible view, with provider_pending set while a provider report is outstanding.
        """
        return ExecutionView(
            id=execution.id,
            pipeline_id=execution.target.pipeline_id,
            models=list(execution.target.models),
            pipeline_digest=execution.target.pipeline_digest,
            status=execution.status,
            configuration=execution.invocation.configuration,
            result=(
                execution.result
                if execution.status is ExecutionStatus.SUCCEEDED
                else None
            ),
            error=(
                execution.error if execution.status is ExecutionStatus.FAILED else None
            ),
            provider_pending=execution.work_kind
            in {
                WorkKind.PROVIDER_FAILURE_REPORT,
                WorkKind.PROVIDER_CANCELLATION_REPORT,
            },
        )
