from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from execution_scheduler.contracts import (
    ExecutionError,
    ExecutionErrorCode,
    ExecutionStatus,
)
from execution_scheduler.domain import (
    ProcessingFailed,
    ProcessingRejected,
    ProcessingSucceeded,
    ProviderFinishFailed,
    ProviderFinishSucceeded,
    StaleWorkClaimError,
    WorkKind,
)


if TYPE_CHECKING:
    from datetime import datetime, timedelta

    from execution_scheduler.domain import ExecutionRecord, RetryPolicy, Settlement


class ExecutionTransitions:
    def __init__(
        self,
        processing_policy: RetryPolicy,
        finish_policy: RetryPolicy,
    ) -> None:
        self._processing_policy = processing_policy
        self._finish_policy = finish_policy

    def settle(
        self,
        execution: ExecutionRecord,
        settlement: Settlement,
        now: datetime,
    ) -> ExecutionRecord:
        if isinstance(settlement, ProcessingSucceeded):
            self._require_kind(execution, WorkKind.PROCESSING)
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
                return self._retry(
                    execution,
                    now,
                    self._finish_policy.backoff(execution.finish_attempts),
                    status=execution.status,
                )
            if execution.work_kind is WorkKind.PROVIDER_FINISH:
                return self._terminal_failure(execution, settlement.error, now)
            return self._clear_work(execution, now)
        raise TypeError("unknown work settlement")

    def recover_expired(
        self,
        execution: ExecutionRecord,
        now: datetime,
    ) -> ExecutionRecord:
        if (
            execution.work_kind is WorkKind.PROCESSING
            and execution.processing_attempts >= self._processing_policy.max_attempts
        ):
            return self._terminal_failure(
                execution,
                ExecutionError(
                    code=ExecutionErrorCode.PROCESSING_ATTEMPTS_EXHAUSTED,
                    message=(
                        "workflow processing did not complete within its retry policy"
                    ),
                ),
                now,
                report=True,
            )
        if execution.finish_attempts >= self._finish_policy.max_attempts:
            if execution.work_kind is WorkKind.PROVIDER_FINISH:
                return self._terminal_failure(
                    execution,
                    ExecutionError(
                        code=ExecutionErrorCode.PROVIDER_FINISH_ATTEMPTS_EXHAUSTED,
                        message=(
                            "provider finish did not complete within its retry policy"
                        ),
                    ),
                    now,
                )
            return self._clear_work(execution, now)
        attempts = (
            execution.processing_attempts
            if execution.work_kind is WorkKind.PROCESSING
            else execution.finish_attempts
        )
        policy = (
            self._processing_policy
            if execution.work_kind is WorkKind.PROCESSING
            else self._finish_policy
        )
        return self._retry(
            execution,
            now,
            policy.backoff(attempts),
            status=(
                ExecutionStatus.QUEUED
                if execution.work_kind is WorkKind.PROCESSING
                else execution.status
            ),
        )

    def deadline(self, execution: ExecutionRecord, now: datetime) -> ExecutionRecord:
        if execution.status in {ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            return self._clear_work(execution, now)
        return self._terminal_failure(
            execution,
            ExecutionError(
                code=ExecutionErrorCode.DEADLINE_EXCEEDED,
                message="execution deadline was exceeded",
            ),
            now,
        )

    @staticmethod
    def _require_kind(execution: ExecutionRecord, expected: WorkKind) -> None:
        if execution.work_kind is not expected:
            raise StaleWorkClaimError(f"claim is not {expected.value}")

    @staticmethod
    def _require_provider_kind(execution: ExecutionRecord) -> None:
        if execution.work_kind not in {
            WorkKind.PROVIDER_FINISH,
            WorkKind.PROVIDER_FAILURE_REPORT,
            WorkKind.PROVIDER_CANCELLATION_REPORT,
        }:
            raise StaleWorkClaimError("claim is not provider work")

    @staticmethod
    def _retry(
        execution: ExecutionRecord,
        now: datetime,
        backoff: timedelta,
        *,
        status: ExecutionStatus,
    ) -> ExecutionRecord:
        return replace(
            execution,
            status=status,
            next_attempt_at=min(now + backoff, execution.deadline_at),
            lease_owner=None,
            fencing_token=None,
            lease_expires_at=None,
            updated_at=now,
        )

    @staticmethod
    def _terminal_failure(
        execution: ExecutionRecord,
        error: ExecutionError,
        now: datetime,
        *,
        report: bool = False,
    ) -> ExecutionRecord:
        return replace(
            execution,
            status=ExecutionStatus.FAILED,
            work_kind=(WorkKind.PROVIDER_FAILURE_REPORT if report else None),
            next_attempt_at=now if report else None,
            lease_owner=None,
            fencing_token=None,
            lease_expires_at=None,
            error=error,
            updated_at=now,
        )

    @staticmethod
    def _clear_work(execution: ExecutionRecord, now: datetime) -> ExecutionRecord:
        return replace(
            execution,
            work_kind=None,
            next_attempt_at=None,
            lease_owner=None,
            fencing_token=None,
            lease_expires_at=None,
            updated_at=now,
        )
