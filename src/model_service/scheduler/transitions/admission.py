"""Admitting and cancelling executions."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from model_service.scheduler.internal_state import (
    Admission,
    ExecutionRecord,
    WorkKind,
)
from model_service.scheduler.status import ExecutionStatus


if TYPE_CHECKING:
    from datetime import datetime


class AdmissionTransitions:
    """Admitting and cancelling executions."""

    @staticmethod
    def admit(
        admission: Admission,
        execution_id: str,
        now: datetime,
    ) -> ExecutionRecord:
        """Create a queued execution with immediate processing due, zero attempts, and no lease or outcome.

        Args:
            admission: Validated submission to admit.
            execution_id: ID for the new execution.
            now: Current database time.

        Returns:
            The new queued record.
        """
        return ExecutionRecord(
            id=execution_id,
            identity=admission.identity,
            builder_id=admission.builder_id,
            model_id=admission.model_id,
            status=ExecutionStatus.QUEUED,
            invocation=admission.invocation,
            protected_context=admission.protected_context,
            deadline_at=admission.deadline_at,
            terminal_report_deadline_at=admission.terminal_report_deadline_at,
            work_kind=WorkKind.PROCESSING,
            next_attempt_at=now,
            processing_attempts=0,
            processing_rejections=0,
            finish_attempts=0,
            finish_mutation_started=False,
            lease_owner=None,
            fencing_token=None,
            lease_expires_at=None,
            result=None,
            error=None,
            created_at=now,
            updated_at=now,
            priority=admission.priority,
        )

    @staticmethod
    def cancel(execution: ExecutionRecord, now: datetime) -> ExecutionRecord:
        """Make a nonterminal execution cancelled, fence its attempt, and queue provider cancellation reporting.

        Publication marks and finish attempts are reset because they belong to the
        publication, not the report.

        Args:
            execution: Current execution record.
            now: Current database time.

        Returns:
            The cancelled record, or the same record unchanged if it already ended.
        """
        if execution.status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        }:
            return execution
        return replace(
            execution,
            status=ExecutionStatus.CANCELLED,
            work_kind=WorkKind.PROVIDER_CANCELLATION_REPORT,
            next_attempt_at=now,
            lease_owner=None,
            fencing_token=None,
            lease_expires_at=None,
            # A publication's mark and attempts are not the report's: left set,
            # recovery would take the report as sent, or as out of attempts.
            finish_mutation_started=False,
            finish_attempts=0,
            updated_at=now,
        )
