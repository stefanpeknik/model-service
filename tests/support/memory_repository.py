"""Implement the execution repository in memory for deterministic local tests.

One asyncio lock serializes operations; an injected clock permits deadline and recovery
tests without real waiting. This adapter does not survive process exit.
"""

from __future__ import annotations

import asyncio
import inspect
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

from model_service.builders.registry import get_builder
from model_service.contracts.invocation import ExecutionContext, Invocation
from model_service.contracts.model_declarations import (
    InferenceContract,
    ModelBinding,
    SpatialScoresModel,
    TileEmbeddingModel,
)
from model_service.contracts.result import BuilderResult
from model_service.scheduler.internal_state import (
    Admission,
    AdmissionResult,
    ExecutionRecord,
    RenewDirective,
    RetryPolicy,
    Settlement,
    WorkClaim,
    WorkKind,
)
from model_service.scheduler.public import (
    ScheduleConflictError,
    UnknownExecutionError,
)
from model_service.scheduler.status import ExecutionStatus
from model_service.scheduler.transitions import ExecutionTransitions
from model_service.scheduler.transitions.common import work_deadline_at


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable, Collection, Mapping

    from model_service.scheduler.public import SubmissionIdentity


def utc_now() -> datetime:
    """Return the timezone-aware UTC wall clock used when no test clock is injected."""
    return datetime.now(UTC)


class InMemoryExecutionRepository:
    """Mirror durable repository semantics using dictionaries and the shared execution state machine."""

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] = utc_now,
        per_model_limit: int = 1,
        processing_policy: RetryPolicy | None = None,
        finish_policy: RetryPolicy | None = None,
    ) -> None:
        """Set capacity and retry policies, create the state machine, and start with empty records under one async lock."""
        if per_model_limit <= 0:
            raise ValueError("per-model limits must be positive")
        self._clock = clock
        self._per_model_limit = per_model_limit
        self._processing_policy = processing_policy or RetryPolicy(
            max_attempts=3,
            initial_backoff=timedelta(seconds=5),
            maximum_backoff=timedelta(minutes=1),
        )
        self._finish_policy = finish_policy or RetryPolicy(
            max_attempts=3,
            initial_backoff=timedelta(seconds=5),
            maximum_backoff=timedelta(minutes=1),
        )
        self._transitions = ExecutionTransitions(
            self._processing_policy,
            self._finish_policy,
        )
        self._executions: dict[str, ExecutionRecord] = {}
        self._submissions: dict[tuple[str, str], str] = {}
        self._lock = asyncio.Lock()

    async def aclose(self) -> None:
        return None

    async def check_ready(self) -> None:
        """Acquire the repository lock to confirm this in-process adapter is available."""
        async with self._operation():
            return

    async def maintain(self) -> None:
        """Expire deadlines and recover abandoned claims under the repository lock."""
        async with self._operation():
            now = self._clock()
            self._expire_deadlines(now)
            self._recover_expired_leases(now)

    async def find_submission(
        self,
        identity: SubmissionIdentity,
    ) -> ExecutionRecord | None:
        """Look up the provider/key pair and reject a replay with a changed fingerprint."""
        async with self._operation():
            execution_id = self._submissions.get((identity.provider, identity.key))
            if execution_id is None:
                return None
            execution = self._executions[execution_id]
            self._check_fingerprint(execution, identity)
            return execution

    async def admit(self, admission: Admission) -> AdmissionResult:
        """Create one execution per provider/key pair or return its identical replay while holding the lock."""
        async with self._operation():
            key = (admission.identity.provider, admission.identity.key)
            existing_id = self._submissions.get(key)
            if existing_id is not None:
                existing = self._executions[existing_id]
                self._check_fingerprint(existing, admission.identity)
                return AdmissionResult(existing, created=False)
            now = self._clock()
            execution = self._transitions.admission.admit(
                admission,
                str(uuid4()),
                now,
            )
            self._executions[execution.id] = execution
            self._submissions[key] = execution.id
            return AdmissionResult(execution, created=True)

    async def get(self, execution_id: str) -> ExecutionRecord:
        """Return a record under the lock, raising UnknownExecutionError if absent."""
        async with self._operation():
            return self._get(execution_id)

    async def get_many(self, execution_ids: Collection[str]) -> list[ExecutionRecord]:
        async with self._operation():
            return [
                self._executions[value]
                for value in execution_ids
                if value in self._executions
            ]

    async def request_cancel(self, execution_id: str) -> ExecutionRecord:
        """Apply cancellation atomically and store the record with its previous claim invalidated."""
        async with self._operation():
            execution = self._get(execution_id)
            cancelled = self._transitions.admission.cancel(execution, self._clock())
            self._store(cancelled)
            return cancelled

    async def claim_due(
        self,
        owner: str,
        lease_duration: timedelta,
    ) -> WorkClaim | None:
        """Lease the highest-priority due work within per-model processing capacity."""
        async with self._operation():
            now = self._clock()
            active = self._active_processing(now)
            due = [
                execution
                for execution in self._executions.values()
                if execution.work_kind is not None
                and execution.next_attempt_at is not None
                and execution.next_attempt_at <= now
                and execution.fencing_token is None
            ]
            first_created: dict[str, datetime] = {}
            top_priority: dict[str, int] = {}
            for execution in due:
                if execution.work_kind is WorkKind.PROCESSING:
                    target = execution.model_id
                    first_created[target] = min(
                        first_created.get(target, execution.created_at),
                        execution.created_at,
                    )
                    top_priority[target] = max(
                        top_priority.get(target, execution.priority),
                        execution.priority,
                    )

            def order(item: ExecutionRecord) -> tuple[object, ...]:
                if item.work_kind is WorkKind.PROCESSING:
                    target = item.model_id
                    group = (-top_priority[target], first_created[target])
                else:
                    group = (-item.priority, item.next_attempt_at)
                return (
                    item.work_kind is WorkKind.PROCESSING,
                    *group,
                    -item.priority,
                    item.next_attempt_at,
                    item.created_at,
                )

            for execution in sorted(due, key=order):
                target = execution.model_id
                if (
                    execution.work_kind is WorkKind.PROCESSING
                    and active.get(target, 0) >= self._per_model_limit
                ):
                    continue
                token = str(uuid4())
                if execution.work_kind is WorkKind.PROCESSING:
                    active[target] = active.get(target, 0) + 1
                claimed = self._transitions.leases.claim(
                    execution,
                    owner=owner,
                    fencing_token=token,
                    lease_duration=lease_duration,
                    now=now,
                )
                self._executions[execution.id] = claimed.execution
                return claimed.claim
            return None

    async def renew(
        self,
        execution_id: str,
        fencing_token: str,
        lease_duration: timedelta,
    ) -> RenewDirective:
        """Apply the shared renewal decision and persist any state change before returning its directive."""
        async with self._operation():
            execution = self._get(execution_id)
            now = self._clock()
            decision = self._transitions.leases.decide_renewal(
                execution,
                fencing_token=fencing_token,
                lease_duration=lease_duration,
                now=now,
            )
            if decision.execution is not None:
                self._store(decision.execution)
            return decision.directive

    async def settle(
        self,
        execution_id: str,
        fencing_token: str,
        settlement: Settlement,
    ) -> ExecutionRecord:
        """Apply a token-checked settlement; persist deadline changes before raising a rejection."""
        async with self._operation():
            execution = self._get(execution_id)
            now = self._clock()
            decision = self._transitions.outcomes.decide_settlement(
                execution,
                fencing_token,
                settlement,
                now,
            )
            if decision.execution is not None:
                self._store(decision.execution)
            return decision.accepted_execution()

    async def mark_finish_mutation_started(
        self,
        execution_id: str,
        fencing_token: str,
    ) -> None:
        """Atomically validate and mark the boundary before a provider's external mutation."""
        async with self._operation():
            execution = self._get(execution_id)
            self._executions[execution_id] = (
                self._transitions.outcomes.mark_finish_mutation_started(
                    execution,
                    fencing_token,
                    self._clock(),
                )
            )

    def _recover_expired_leases(self, now: datetime) -> None:
        """Replace each abandoned leased record with its shared recovery decision."""
        for execution in tuple(self._executions.values()):
            if (
                execution.fencing_token is None
                or execution.lease_expires_at is None
                or execution.lease_expires_at > now
            ):
                continue
            self._store(self._transitions.expiry.recover_expired(execution, now))

    @asynccontextmanager
    async def _operation(self) -> AsyncIterator[None]:
        async with self._lock:
            yield

    def _store(self, execution: ExecutionRecord) -> None:
        self._executions[execution.id] = execution

    def _active_processing(self, now: datetime) -> dict[str, int]:
        """Count only processing claims with an unexpired lease per model."""
        counts: dict[str, int] = {}
        for execution in self._executions.values():
            if (
                execution.work_kind is WorkKind.PROCESSING
                and execution.fencing_token is not None
                and execution.lease_expires_at is not None
                and execution.lease_expires_at > now
            ):
                target = execution.model_id
                counts[target] = counts.get(target, 0) + 1
        return counts

    def _expire_deadlines(self, now: datetime) -> None:
        """Apply the appropriate execution or reporting deadline to records that still have work."""
        for execution in tuple(self._executions.values()):
            deadline_at = work_deadline_at(execution)
            if deadline_at <= now and (
                execution.status
                not in {
                    ExecutionStatus.SUCCEEDED,
                    ExecutionStatus.FAILED,
                    ExecutionStatus.CANCELLED,
                }
                or execution.work_kind is not None
            ):
                self._store(self._transitions.expiry.deadline(execution, now))

    @staticmethod
    def _check_fingerprint(
        execution: ExecutionRecord,
        identity: SubmissionIdentity,
    ) -> None:
        """Reject different request contents under the same provider submission key."""
        if execution.identity.fingerprint != identity.fingerprint:
            raise ScheduleConflictError(
                "submission identity was reused with a different request"
            )

    def _get(self, execution_id: str) -> ExecutionRecord:
        """Resolve an execution ID or translate a dictionary miss into UnknownExecutionError."""
        try:
            return self._executions[execution_id]
        except KeyError as error:
            raise UnknownExecutionError(execution_id) from error


def resolve_test_model(builder_id: str, model_id: str) -> ModelBinding:
    """Resolve any model ID to a small binding of the contract the builder accepts."""
    spec = get_builder(builder_id)
    if spec.accepts is InferenceContract.SPATIAL_SCORES:
        return SpatialScoresModel(
            id=model_id, tile_size=8, output_tile_size=1, n_channels=1, mpp=0.5
        )
    return TileEmbeddingModel(id=model_id, tile_size=8, mpp=0.5)


type Handler = Callable[
    [Invocation, ExecutionContext], BuilderResult | Awaitable[BuilderResult]
]


class FakeDispatch:
    """Run in-process handlers by builder ID and record the models they received."""

    def __init__(self, handlers: Mapping[str, Handler]) -> None:
        self._handlers = dict(handlers)
        self.models: list[ModelBinding] = []

    async def run(
        self,
        builder_id: str,
        invocation: Invocation,
        model: ModelBinding,
        execution_context: ExecutionContext,
    ) -> BuilderResult:
        self.models.append(model)
        value = self._handlers[builder_id](invocation, execution_context)
        if inspect.isawaitable(value):
            return await value
        return value
