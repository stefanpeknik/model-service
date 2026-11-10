"""Settling an attempt's outcome: success, retry or failure."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from model_service.scheduler.execution_records import (
    ExecutionRecord,
    InferenceSucceeded,
    ProcessingFailed,
    ProcessingRejected,
    ProcessingSucceeded,
    ProviderFinishFailed,
    ProviderFinishSucceeded,
    StaleWorkClaimError,
    WorkKind,
)
from model_service.scheduler.executions import ExecutionError
from model_service.scheduler.status import (
    ExecutionErrorCode,
    ExecutionStatus,
)


if TYPE_CHECKING:
    from datetime import datetime

    from model_service.scheduler.execution_records import Settlement


from model_service.scheduler.transitions.base import (
    SettlementAccepted,
    SettlementDecision,
    SettlementRejected,
    SettlementRejection,
    _has_provider_phase,
)
from model_service.scheduler.transitions.recovery import RecoveryTransitions


class SettlementTransitions(RecoveryTransitions):
    """Settling an attempt's outcome: success, retry or failure."""

    def decide_settlement(
        self,
        execution: ExecutionRecord,
        fencing_token: str,
        settlement: Settlement,
        now: datetime,
    ) -> SettlementDecision:
        """Check token, deadline, and lease before applying an outcome.

        A deadline update is kept even when the settlement is rejected, so the caller
        persists it before reporting the rejection.

        Args:
            execution: Current execution record.
            fencing_token: Token of the settling attempt.
            settlement: Outcome of the attempt.
            now: Current database time.

        Returns:
            The state update and the rejection reason, if any.

        Raises:
            StaleWorkClaimError: The outcome does not fit the claimed work kind or operation.
            TypeError: The settlement type is unknown.
        """
        if execution.fencing_token != fencing_token:
            return SettlementRejected(SettlementRejection.STALE_CLAIM)
        if self.work_deadline_at(execution) <= now:
            return SettlementRejected(
                SettlementRejection.DEADLINE_EXCEEDED, self.deadline(execution, now)
            )
        if execution.lease_expires_at is None or execution.lease_expires_at <= now:
            return SettlementRejected(SettlementRejection.STALE_CLAIM)
        return SettlementAccepted(self.settle(execution, settlement, now))

    def mark_finish_mutation_started(
        self,
        execution: ExecutionRecord,
        fencing_token: str,
        now: datetime,
    ) -> ExecutionRecord:
        """Validate a live provider claim and mark that an external write is about to begin.

        After the mark, an interrupted publication becomes outcome-unknown instead of being retried.

        Args:
            execution: Current execution record.
            fencing_token: Token of the provider attempt.
            now: Current database time.

        Returns:
            The record with finish_mutation_started set.

        Raises:
            StaleWorkClaimError: The claim is not provider work, its token or lease is no longer current, or its deadline passed.
        """
        self._require_provider_kind(execution)
        if (
            execution.fencing_token != fencing_token
            or execution.lease_expires_at is None
            or execution.lease_expires_at <= now
            or self.work_deadline_at(execution) <= now
        ):
            raise StaleWorkClaimError("provider finish claim is no longer current")
        return replace(execution, finish_mutation_started=True, updated_at=now)

    def settle(
        self,
        execution: ExecutionRecord,
        settlement: Settlement,
        now: datetime,
    ) -> ExecutionRecord:
        """Advance processing or provider work according to its outcome and retry policy.

        Backpressure rejections do not count as attempts. A publication that failed or
        whose outcome is unknown queues a failure report, since an EMPAIA job must not
        stay open.

        Args:
            execution: Current execution record.
            settlement: Outcome of the attempt.
            now: Current database time.

        Returns:
            The record after the outcome.

        Raises:
            StaleWorkClaimError: The outcome does not fit the work kind, or a builder and
                raw inference settle each other's result type.
            TypeError: The settlement type is unknown.
        """
        if isinstance(settlement, InferenceSucceeded):
            self._require_kind(execution, WorkKind.PROCESSING)
            if _has_provider_phase(execution):
                raise StaleWorkClaimError("a builder cannot settle a model response")
            return replace(
                execution,
                status=ExecutionStatus.SUCCEEDED,
                result=settlement.result,
                work_kind=None,
                next_attempt_at=None,
                lease_owner=None,
                fencing_token=None,
                lease_expires_at=None,
                updated_at=now,
            )
        if isinstance(settlement, ProcessingSucceeded):
            self._require_kind(execution, WorkKind.PROCESSING)
            if not _has_provider_phase(execution):
                raise StaleWorkClaimError(
                    "raw inference cannot settle a builder result"
                )
            return replace(
                execution,
                status=ExecutionStatus.RUNNING,
                result=settlement.result,
                work_kind=WorkKind.PROVIDER_FINISH,
                next_attempt_at=now,
                lease_owner=None,
                fencing_token=None,
                lease_expires_at=None,
                updated_at=now,
            )
        if isinstance(settlement, ProcessingRejected):
            self._require_kind(execution, WorkKind.PROCESSING)
            rejection_count = execution.processing_rejections + 1
            backoff = self._processing_policy.backoff(rejection_count)
            if settlement.retry_after is not None:
                backoff = max(backoff, settlement.retry_after)
            return replace(
                execution,
                status=ExecutionStatus.QUEUED,
                processing_attempts=execution.processing_attempts - 1,
                processing_rejections=rejection_count,
                next_attempt_at=min(now + backoff, execution.deadline_at),
                lease_owner=None,
                fencing_token=None,
                lease_expires_at=None,
                updated_at=now,
            )
        if isinstance(settlement, ProcessingFailed):
            self._require_kind(execution, WorkKind.PROCESSING)
            if (
                settlement.retryable
                and execution.processing_attempts < self._processing_policy.max_attempts
            ):
                return self._retry(
                    execution,
                    now,
                    self._processing_policy.backoff(execution.processing_attempts),
                    status=ExecutionStatus.QUEUED,
                )
            return self._terminal_failure(
                execution,
                settlement.error,
                now,
                report=True,
            )
        if isinstance(settlement, ProviderFinishSucceeded):
            self._require_provider_kind(execution)
            return replace(
                execution,
                status=(
                    ExecutionStatus.SUCCEEDED
                    if execution.work_kind is WorkKind.PROVIDER_FINISH
                    else execution.status
                ),
                work_kind=None,
                next_attempt_at=None,
                lease_owner=None,
                fencing_token=None,
                lease_expires_at=None,
                updated_at=now,
            )
        if isinstance(settlement, ProviderFinishFailed):
            self._require_provider_kind(execution)
            if (
                settlement.retryable
                and not settlement.outcome_unknown
                and execution.finish_attempts < self._finish_policy.max_attempts
            ):
                return replace(
                    self._retry(
                        execution,
                        now,
                        self._finish_policy.backoff(execution.finish_attempts),
                        status=execution.status,
                    ),
                    finish_mutation_started=False,
                )
            if execution.work_kind is WorkKind.PROVIDER_FINISH:
                # Tell the provider: an EMPAIA job must not stay open after a
                # publication that failed or whose outcome is unknown.
                return self._terminal_failure(
                    execution, settlement.error, now, report=True
                )
            return self._clear_work(execution, now)
        raise TypeError("unknown work settlement")

    def unreadable(self, execution: ExecutionRecord, now: datetime) -> ExecutionRecord:
        """End an execution whose provider context cannot be decrypted; nothing can report it.

        One that already ended, and only had a report due, keeps its outcome.

        Args:
            execution: Current execution record.
            now: Current database time.

        Returns:
            The failed record with no pending work, or the ended record with its work cleared.
        """
        if execution.status in {ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            return self._clear_work(execution, now)
        return self._terminal_failure(
            execution,
            ExecutionError(
                code=ExecutionErrorCode.PROTECTED_CONTEXT_UNREADABLE,
                message="the execution's stored provider context cannot be read",
            ),
            now,
        )
