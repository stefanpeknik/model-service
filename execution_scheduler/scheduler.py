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

from execution_scheduler import inference
from execution_scheduler.active_work import ActiveWorkRegistry
from execution_scheduler.contracts import (
    ExecutionStatus,
    ExecutionView,
    InferenceInvocation,
    InferenceResult,
    InferenceScheduleRequest,
    InvalidScheduleRequestError,
    Operation,
    ResultUnavailableError,
    ScheduleRequest,
    ScheduleResult,
    SchedulingUnavailableError,
)
from execution_scheduler.domain import (
    Admission,
    ExecutionRecord,
    SubmissionIdentity,
)
from execution_scheduler.repository import ExecutionRepositoryUnavailableError
from execution_scheduler.worker import ExecutionWorker


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Mapping

    from builders.client import BuilderDispatch
    from execution_scheduler.contracts import ModelResolver
    from execution_scheduler.inference import InferenceTarget
    from execution_scheduler.providers import ProviderAdapter
    from execution_scheduler.repository import ExecutionRepository


class ExecutionScheduler:
    """Admission, status, cancellation and the background worker's lifetime."""

    def __init__(
        self,
        *,
        repository: ExecutionRepository,
        dispatch: BuilderDispatch,
        providers: Mapping[str, ProviderAdapter],
        resolve_model: ModelResolver,
        default_timeout: timedelta,
        maximum_timeout: timedelta,
        clock: Callable[[], datetime] | None = None,
        lease_duration: timedelta = timedelta(seconds=60),
        poll_interval: float = 0.5,
        max_concurrent: int = 1,
        inference_target: InferenceTarget | None = None,
        max_concurrent_inference: int = 0,
    ) -> None:
        """Validate timing settings and wire a worker with a private local cancellation registry. No tasks start here."""
        if lease_duration <= timedelta(0):
            raise ValueError("lease_duration must be positive")
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        self._repository = repository
        self._providers = dict(providers)
        self._resolve_model = resolve_model
        self._default_timeout = default_timeout
        self._maximum_timeout = maximum_timeout
        self._clock = clock or (lambda: datetime.now(UTC))
        self._inference = inference_target
        self._active_work = ActiveWorkRegistry()
        self._worker = ExecutionWorker(
            repository=repository,
            providers=self._providers,
            dispatch=dispatch,
            resolve_model=resolve_model,
            owner=f"execution-scheduler-{uuid4()}",
            lease_duration=lease_duration,
            active_work=self._active_work,
            max_concurrent=max_concurrent,
            inference_target=inference_target,
            max_concurrent_inference=max_concurrent_inference,
        )
        self._poll_interval = poll_interval
        self._running = False

    @asynccontextmanager
    async def running(self) -> AsyncIterator[None]:
        """Start and join worker tasks for this context's lifetime."""
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
        """Translate repository readiness failures into a public scheduling availability error."""
        try:
            await self._repository.check_ready()
        except ExecutionRepositoryUnavailableError as error:
            raise SchedulingUnavailableError(
                "execution repository is temporarily unavailable"
            ) from error

    async def schedule(self, request: ScheduleRequest) -> ScheduleResult:
        """Identify and replay a submission, prepare new inputs, then admit the complete execution."""
        self._resolve_model(request.builder_id, request.model_id)
        try:
            provider = self._providers[request.provider]
        except KeyError as error:
            raise InvalidScheduleRequestError("unknown provider") from error
        identity = provider.identify(request)
        if (existing := await self._find(identity)) is not None:
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
        return await self._admit(
            Admission(
                identity=identity,
                builder_id=request.builder_id,
                model_id=request.model_id,
                invocation=invocation,
                protected_context=prepared.protected_context,
                deadline_at=deadline_at,
                terminal_report_deadline_at=(prepared.terminal_report_deadline_at),
            )
        )

    async def schedule_inference(
        self, request: InferenceScheduleRequest
    ) -> ScheduleResult:
        """Check the payload against the model, then store it with the execution.

        Raises UnknownModelError or IncompatibleInputError before admission.
        """
        if self._inference is None:
            raise SchedulingUnavailableError("scheduled inference is not configured")
        self._inference.check(request.model_id, request.payload)
        identity = inference.identify(request)
        if (existing := await self._find(identity)) is not None:
            return ScheduleResult(self._view(existing), created=False)
        deadline_at = self._clock() + self._requested_timeout(
            request.requested_timeout_seconds
        )
        return await self._admit(
            Admission(
                identity=identity,
                builder_id=None,
                model_id=request.model_id,
                invocation=InferenceInvocation(
                    headers=request.headers, deadline_at=deadline_at
                ),
                protected_context={},
                deadline_at=deadline_at,
                operation=Operation.INFERENCE,
                payload=request.payload,
            )
        )

    async def _find(self, identity: SubmissionIdentity) -> ExecutionRecord | None:
        try:
            return await self._repository.find_submission(identity)
        except ExecutionRepositoryUnavailableError as error:
            raise SchedulingUnavailableError(
                "execution scheduling is temporarily unavailable"
            ) from error

    async def _admit(self, admission: Admission) -> ScheduleResult:
        try:
            admitted = await self._repository.admit(admission)
        except ExecutionRepositoryUnavailableError as error:
            raise SchedulingUnavailableError(
                "execution scheduling is temporarily unavailable"
            ) from error
        return ScheduleResult(self._view(admitted.execution), created=admitted.created)

    async def get(self, execution_id: str) -> ExecutionView:
        """Read a durable record and project only its caller-visible status and terminal outcome."""
        try:
            execution = await self._repository.get(execution_id)
        except ExecutionRepositoryUnavailableError as error:
            raise SchedulingUnavailableError(
                "execution repository is temporarily unavailable"
            ) from error
        return self._view(execution)

    async def get_many(self, execution_ids: list[str]) -> list[ExecutionView]:
        """Views of the executions that exist, in the requested order."""
        try:
            executions = await self._repository.get_many(execution_ids)
        except ExecutionRepositoryUnavailableError as error:
            raise SchedulingUnavailableError(
                "execution repository is temporarily unavailable"
            ) from error
        return [self._view(execution) for execution in executions]

    async def inference_result(
        self, execution_id: str
    ) -> tuple[InferenceResult, bytes]:
        """Raise UnknownExecutionError or ResultUnavailableError when there is none."""
        try:
            execution = await self._repository.get(execution_id)
            if not isinstance(execution.result, InferenceResult):
                raise ResultUnavailableError("execution has no inference response")
            content = await self._repository.inference_content(execution_id)
        except ExecutionRepositoryUnavailableError as error:
            raise SchedulingUnavailableError(
                "execution repository is temporarily unavailable"
            ) from error
        return execution.result, content

    async def cancel(self, execution_id: str) -> ExecutionView:
        """Record cancellation and fence the current claim before requesting cancellation of any local task."""
        try:
            execution = await self._repository.request_cancel(execution_id)
        except ExecutionRepositoryUnavailableError as error:
            raise SchedulingUnavailableError(
                "execution repository is temporarily unavailable"
            ) from error
        self._active_work.cancel(execution_id)
        return self._view(execution)

    def _requested_timeout(self, seconds: int | None) -> timedelta:
        """Use the default timeout or clamp a positive requested timeout to the configured maximum."""
        if seconds is None:
            return self._default_timeout
        if seconds <= 0:
            raise InvalidScheduleRequestError("requested timeout must be positive")
        return min(timedelta(seconds=seconds), self._maximum_timeout)

    @staticmethod
    def _view(execution: ExecutionRecord) -> ExecutionView:
        """Hide internal fields and unfinished results when converting a record to the public execution view."""
        return ExecutionView(
            id=execution.id,
            operation=execution.operation,
            builder_id=execution.builder_id,
            model_id=execution.model_id,
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
