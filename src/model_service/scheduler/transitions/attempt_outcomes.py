"""Settling an attempt's outcome: success, retry or failure."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from model_service.scheduler.internal_state import (
    ExecutionRecord,
    ProcessingFailed,
    ProcessingRejected,
    ProcessingSucceeded,
    ProviderFinishFailed,
    ProviderFinishSucceeded,
    StaleWorkClaimError,
    WorkKind,
)
from model_service.scheduler.public import ExecutionError
from model_service.scheduler.status import (
    ExecutionErrorCode,
    ExecutionStatus,
)


if TYPE_CHECKING:
    from datetime import datetime

    from model_service.scheduler.internal_state import RetryPolicy, Settlement
    from model_service.scheduler.transitions.expiry import ExpiryTransitions


from model_service.scheduler.transitions.common import (
    SettlementAccepted,
    SettlementDecision,
    SettlementRejected,
    SettlementRejection,
    clear_work,
    require_kind,
    require_provider_kind,
    retry,
    terminal_failure,
    work_deadline_at,
)


class AttemptOutcomeTransitions:
    """Settling an attempt's outcome: success, retry or failure."""

    def __init__(
        self,
        processing_policy: RetryPolicy,
        finish_policy: RetryPolicy,
        expiry: ExpiryTransitions,
    ) -> None:
        """Keep the retry policies and the expiry rules a settlement past its deadline applies.

        Args:
            processing_policy: Attempts and backoff for builder processing.
            finish_policy: Attempts and backoff for provider finish and reports.
            expiry: Deadline and lease-recovery rules.
        """
        self._processing_policy = processing_policy
        self._finish_policy = finish_policy
        self._expiry = expiry

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
            StaleWorkClaimError: The outcome does not fit the claimed work kind.
            TypeError: The settlement type is unknown.
        """
        if execution.fencing_token != fencing_token:
            return SettlementRejected(SettlementRejection.STALE_CLAIM)
        if work_deadline_at(execution) <= now:
            return SettlementRejected(
                SettlementRejection.DEADLINE_EXCEEDED,
                self._expiry.deadline(execution, now),
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
        require_provider_kind(execution)
        if (
            execution.fencing_token != fencing_token
            or execution.lease_expires_at is None
            or execution.lease_expires_at <= now
            or work_deadline_at(execution) <= now
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
        whose outcome is unknown queues a failure report, since an external job must not
        stay open.

        Args:
            execution: Current execution record.
            settlement: Outcome of the attempt.
            now: Current database time.

        Returns:
            The record after the outcome.

        Raises:
            StaleWorkClaimError: The outcome does not fit the work kind.
            TypeError: The settlement type is unknown.
        """
        if isinstance(settlement, ProcessingSucceeded):
            require_kind(execution, WorkKind.PROCESSING)
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
            require_kind(execution, WorkKind.PROCESSING)
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
            require_kind(execution, WorkKind.PROCESSING)
            if (
                settlement.retryable
                and execution.processing_attempts < self._processing_policy.max_attempts
            ):
                return retry(
                    execution,
                    now,
                    self._processing_policy.backoff(execution.processing_attempts),
                    status=ExecutionStatus.QUEUED,
                )
            return terminal_failure(
                execution,
                settlement.error,
                now,
                report=True,
            )
        if isinstance(settlement, ProviderFinishSucceeded):
            require_provider_kind(execution)
            return replace(
                execution,
                status=(
                    ExecutionStatus.SUCCEEDED
                    if execution.work_kind is WorkKind.PROVIDER_FINISH
                    else execution.status
                ),
                result=settlement.result or execution.result,
                work_kind=None,
                next_attempt_at=None,
                lease_owner=None,
                fencing_token=None,
                lease_expires_at=None,
                updated_at=now,
            )
        if isinstance(settlement, ProviderFinishFailed):
            require_provider_kind(execution)
            if (
                settlement.retryable
                and not settlement.outcome_unknown
                and execution.finish_attempts < self._finish_policy.max_attempts
            ):
                return replace(
                    retry(
                        execution,
                        now,
                        self._finish_policy.backoff(execution.finish_attempts),
                        status=execution.status,
                    ),
                    finish_mutation_started=False,
                )
            if execution.work_kind is WorkKind.PROVIDER_FINISH:
                # Tell the provider: an external job must not stay open after a
                # publication that failed or whose outcome is unknown.
                return terminal_failure(execution, settlement.error, now, report=True)
            return clear_work(execution, now)
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
            return clear_work(execution, now)
        return terminal_failure(
            execution,
            ExecutionError(
                code=ExecutionErrorCode.PROTECTED_CONTEXT_UNREADABLE,
                message="the execution's stored provider context cannot be read",
            ),
            now,
        )
