from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4

from execution_scheduler.contracts import (
    ExecutionStatus,
    ScheduleConflictError,
    UnknownExecutionError,
)
from execution_scheduler.domain import (
    Admission,
    AdmissionResult,
    ExecutionRecord,
    RenewDirective,
    RetryPolicy,
    Settlement,
    SubmissionIdentity,
    WorkClaim,
    WorkKind,
)
from execution_scheduler.transitions import ExecutionTransitions


if TYPE_CHECKING:
    from collections.abc import Callable


def utc_now() -> datetime:
    return datetime.now(UTC)


class ExecutionRepository(Protocol):
    async def check_ready(self) -> None: ...

    async def maintain(self) -> None: ...

    async def find_submission(
        self,
        identity: SubmissionIdentity,
    ) -> ExecutionRecord | None: ...

    async def admit(self, admission: Admission) -> AdmissionResult: ...

    async def get(self, execution_id: str) -> ExecutionRecord: ...

    async def request_cancel(self, execution_id: str) -> ExecutionRecord: ...

    async def claim_due(
        self,
        owner: str,
        lease_duration: timedelta,
    ) -> WorkClaim | None: ...

    async def renew(
        self,
        execution_id: str,
        fencing_token: str,
        lease_duration: timedelta,
    ) -> RenewDirective: ...

    async def settle(
        self,
        execution_id: str,
        fencing_token: str,
        settlement: Settlement,
    ) -> ExecutionRecord: ...

    async def mark_finish_mutation_started(
        self,
        execution_id: str,
        fencing_token: str,
    ) -> None: ...


class ExecutionRepositoryUnavailableError(RuntimeError):
    pass


class InMemoryExecutionRepository:
    def __init__(
        self,
        *,
        clock: Callable[[], datetime] = utc_now,
        per_workflow_limit: int = 1,
        processing_policy: RetryPolicy | None = None,
        finish_policy: RetryPolicy | None = None,
    ) -> None:
        if per_workflow_limit <= 0:
            raise ValueError("per_workflow_limit must be positive")
        self._clock = clock
        self._per_workflow_limit = per_workflow_limit
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

    async def check_ready(self) -> None:
        async with self._lock:
            return

    async def maintain(self) -> None:
        async with self._lock:
            now = self._clock()
            self._expire_deadlines(now)
            self._recover_expired_leases(now)

    async def find_submission(
        self,
        identity: SubmissionIdentity,
    ) -> ExecutionRecord | None:
        async with self._lock:
            execution_id = self._submissions.get((identity.provider, identity.key))
            if execution_id is None:
                return None
            execution = self._executions[execution_id]
            self._check_fingerprint(execution, identity)
            return execution

    async def admit(self, admission: Admission) -> AdmissionResult:
        async with self._lock:
            key = (admission.identity.provider, admission.identity.key)
            existing_id = self._submissions.get(key)
            if existing_id is not None:
                existing = self._executions[existing_id]
                self._check_fingerprint(existing, admission.identity)
                return AdmissionResult(existing, created=False)
            now = self._clock()
            execution = self._transitions.admit(
                admission,
                str(uuid4()),
                now,
            )
            self._executions[execution.id] = execution
            self._submissions[key] = execution.id
            return AdmissionResult(execution, created=True)

    async def get(self, execution_id: str) -> ExecutionRecord:
        async with self._lock:
            return self._get(execution_id)

    async def request_cancel(self, execution_id: str) -> ExecutionRecord:
        async with self._lock:
            execution = self._get(execution_id)
            cancelled = self._transitions.cancel(execution, self._clock())
            self._executions[execution_id] = cancelled
            return cancelled

    async def claim_due(
        self,
        owner: str,
        lease_duration: timedelta,
    ) -> WorkClaim | None:
        async with self._lock:
            now = self._clock()
            self._expire_deadlines(now)
            self._recover_expired_leases(now)
            active_by_workflow = self._active_processing_by_workflow(now)
            due = [
                execution
                for execution in self._executions.values()
                if execution.work_kind is not None
                and execution.next_attempt_at is not None
                and execution.next_attempt_at <= now
                and execution.fencing_token is None
            ]
            first_created: dict[str, datetime] = {}
            for execution in due:
                if execution.work_kind is WorkKind.PROCESSING:
                    first_created[execution.workflow_id] = min(
                        first_created.get(execution.workflow_id, execution.created_at),
                        execution.created_at,
                    )
            for execution in sorted(
                due,
                key=lambda item: (
                    item.work_kind is WorkKind.PROCESSING,
                    first_created[item.workflow_id]
                    if item.work_kind is WorkKind.PROCESSING
                    else item.next_attempt_at,
                    item.next_attempt_at,
                    item.created_at,
                ),
            ):
                if (
                    execution.work_kind is WorkKind.PROCESSING
                    and active_by_workflow.get(execution.workflow_id, 0)
                    >= self._per_workflow_limit
                ):
                    continue
                token = str(uuid4())
                if execution.work_kind is WorkKind.PROCESSING:
                    active_by_workflow[execution.workflow_id] = (
                        active_by_workflow.get(execution.workflow_id, 0) + 1
                    )
                claimed = self._transitions.claim(
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
        async with self._lock:
            execution = self._get(execution_id)
            now = self._clock()
            decision = self._transitions.decide_renewal(
                execution,
                fencing_token=fencing_token,
                lease_duration=lease_duration,
                now=now,
            )
            if decision.execution is not None:
                self._executions[execution_id] = decision.execution
            return decision.directive

    async def settle(
        self,
        execution_id: str,
        fencing_token: str,
        settlement: Settlement,
    ) -> ExecutionRecord:
        async with self._lock:
            execution = self._get(execution_id)
            now = self._clock()
            decision = self._transitions.decide_settlement(
                execution,
                fencing_token,
                settlement,
                now,
            )
            if decision.execution is not None:
                self._executions[execution_id] = decision.execution
            return decision.accepted_execution()

    async def mark_finish_mutation_started(
        self,
        execution_id: str,
        fencing_token: str,
    ) -> None:
        async with self._lock:
            execution = self._get(execution_id)
            self._executions[execution_id] = (
                self._transitions.mark_finish_mutation_started(
                    execution,
                    fencing_token,
                    self._clock(),
                )
            )

    def _recover_expired_leases(self, now: datetime) -> None:
        for execution_id, execution in tuple(self._executions.items()):
            if (
                execution.fencing_token is None
                or execution.lease_expires_at is None
                or execution.lease_expires_at > now
            ):
                continue
            recovered = self._transitions.recover_expired(execution, now)
            self._executions[execution_id] = recovered

    def _active_processing_by_workflow(self, now: datetime) -> dict[str, int]:
        counts: dict[str, int] = {}
        for execution in self._executions.values():
            if (
                execution.work_kind is WorkKind.PROCESSING
                and execution.fencing_token is not None
                and execution.lease_expires_at is not None
                and execution.lease_expires_at > now
            ):
                counts[execution.workflow_id] = counts.get(execution.workflow_id, 0) + 1
        return counts

    def _expire_deadlines(self, now: datetime) -> None:
        for execution_id, execution in tuple(self._executions.items()):
            deadline_at = self._transitions.work_deadline_at(execution)
            if deadline_at <= now and (
                execution.status
                not in {
                    ExecutionStatus.SUCCEEDED,
                    ExecutionStatus.FAILED,
                    ExecutionStatus.CANCELLED,
                }
                or execution.work_kind is not None
            ):
                self._executions[execution_id] = self._transitions.deadline(
                    execution,
                    now,
                )

    @staticmethod
    def _check_fingerprint(
        execution: ExecutionRecord,
        identity: SubmissionIdentity,
    ) -> None:
        if execution.identity.fingerprint != identity.fingerprint:
            raise ScheduleConflictError(
                "submission identity was reused with a different request"
            )

    def _get(self, execution_id: str) -> ExecutionRecord:
        try:
            return self._executions[execution_id]
        except KeyError as error:
            raise UnknownExecutionError(execution_id) from error
