"""Expired leases and passed deadlines."""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_service.scheduler.internal_state import (
    ExecutionRecord,
    WorkKind,
)
from model_service.scheduler.public import ExecutionError
from model_service.scheduler.status import (
    ExecutionErrorCode,
    ExecutionStatus,
)


if TYPE_CHECKING:
    from datetime import datetime


from model_service.scheduler.transitions.common import (
    TransitionsBase,
)


class ExpiryTransitions(TransitionsBase):
    """Expired leases and passed deadlines."""

    def recover_expired(
        self,
        execution: ExecutionRecord,
        now: datetime,
    ) -> ExecutionRecord:
        """Recover abandoned work through retry or terminal failure.

        Respects attempt limits; a publication interrupted after its mark becomes
        outcome-unknown, and a report interrupted after its mark is taken as sent.

        Args:
            execution: Current execution record.
            now: Current database time.

        Returns:
            The recovered record.
        """
        if (
            execution.work_kind is WorkKind.PROVIDER_FINISH
            and execution.finish_mutation_started
        ):
            return self._finish_outcome_unknown(execution, now)
        if (
            execution.work_kind
            in {
                WorkKind.PROVIDER_FAILURE_REPORT,
                WorkKind.PROVIDER_CANCELLATION_REPORT,
            }
            and execution.finish_mutation_started
        ):
            return self._clear_work(execution, now)
        if (
            execution.work_kind is WorkKind.PROCESSING
            and execution.processing_attempts >= self._processing_policy.max_attempts
        ):
            return self._terminal_failure(
                execution,
                ExecutionError(
                    code=ExecutionErrorCode.PROCESSING_ATTEMPTS_EXHAUSTED,
                    message=(
                        "builder processing did not complete within its retry policy"
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
                    report=True,
                )
            return self._clear_work(execution, now)
        processing = execution.work_kind is WorkKind.PROCESSING
        policy = self._processing_policy if processing else self._finish_policy
        return self._retry(
            execution,
            now,
            policy.backoff(execution.current_attempt),
            status=ExecutionStatus.QUEUED if processing else execution.status,
        )

    def deadline(self, execution: ExecutionRecord, now: datetime) -> ExecutionRecord:
        """Fail overdue active work or stop overdue terminal reporting.

        Interrupted publication becomes outcome-unknown instead. A report whose own
        deadline has not passed is left as it is.

        Args:
            execution: Current execution record.
            now: Current database time.

        Returns:
            The record after the deadline.
        """
        if (
            execution.work_kind is WorkKind.PROVIDER_FINISH
            and execution.finish_mutation_started
        ):
            return self._finish_outcome_unknown(execution, now)
        if execution.status in {ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}:
            if (
                execution.terminal_report_deadline_at is not None
                and execution.terminal_report_deadline_at > now
            ):
                return execution
            return self._clear_work(execution, now)
        return self._terminal_failure(
            execution,
            ExecutionError(
                code=ExecutionErrorCode.DEADLINE_EXCEEDED,
                message="execution deadline was exceeded",
            ),
            now,
            report=(
                execution.terminal_report_deadline_at is not None
                and execution.terminal_report_deadline_at > now
            ),
        )
