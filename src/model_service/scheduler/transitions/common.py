"""The decisions transitions return, and the state and helpers every group shares."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING, Never

from model_service.scheduler.internal_state import (
    ExecutionRecord,
    RenewDirective,
    StaleWorkClaimError,
    WorkClaim,
    WorkKind,
)
from model_service.scheduler.public import ExecutionError
from model_service.scheduler.status import (
    ExecutionErrorCode,
    ExecutionStatus,
)


if TYPE_CHECKING:
    from datetime import datetime, timedelta

    from model_service.scheduler.internal_state import RetryPolicy


@dataclass(frozen=True, slots=True)
class ClaimDecision:
    """Pair the new leased record with the attempt description returned to its worker."""

    execution: ExecutionRecord
    claim: WorkClaim


@dataclass(frozen=True, slots=True)
class RenewalDecision:
    """Return an optional state update and the instruction a lease-renewing worker must follow."""

    execution: ExecutionRecord | None
    directive: RenewDirective


class SettlementRejection(StrEnum):
    """Explain why an attempt's outcome cannot be accepted into durable state."""

    STALE_CLAIM = "work claim is no longer current"
    DEADLINE_EXCEEDED = "execution deadline was exceeded"


@dataclass(frozen=True, slots=True)
class SettlementAccepted:
    """An accepted outcome and the state it leads to."""

    execution: ExecutionRecord

    def accepted_execution(self) -> ExecutionRecord:
        """Return the record after the settlement, once it is persisted."""
        return self.execution


@dataclass(frozen=True, slots=True)
class SettlementRejected:
    """A refused outcome, with any deadline change that must still be persisted."""

    rejection: SettlementRejection
    # Set when the rejection itself changes state (a passed deadline).
    execution: ExecutionRecord | None = None

    def accepted_execution(self) -> Never:
        """Refuse, once any required state update is persisted.

        Raises:
            StaleWorkClaimError: Always; the settlement was stale or past its deadline.
        """
        raise StaleWorkClaimError(self.rejection.value)


type SettlementDecision = SettlementAccepted | SettlementRejected


class TransitionsBase(ABC):
    """Retry policies and the helpers every group of transitions shares.

    Expiry (expiry.py) implements recover_expired and deadline; leases and
    attempt outcomes call them through this base.
    """

    @abstractmethod
    def recover_expired(
        self, execution: ExecutionRecord, now: datetime
    ) -> ExecutionRecord:
        """Recover abandoned work through retry or terminal failure.

        Args:
            execution: The execution whose lease expired.
            now: Database time.

        Returns:
            The recovered execution.
        """

    @abstractmethod
    def deadline(self, execution: ExecutionRecord, now: datetime) -> ExecutionRecord:
        """Fail overdue active work or stop overdue terminal reporting.

        Args:
            execution: The execution to check.
            now: Database time.

        Returns:
            The execution, failed or stopped if its deadline passed.
        """

    def __init__(
        self,
        processing_policy: RetryPolicy,
        finish_policy: RetryPolicy,
    ) -> None:
        self._processing_policy = processing_policy
        self._finish_policy = finish_policy

    @staticmethod
    def _finish_outcome_unknown(
        execution: ExecutionRecord,
        now: datetime,
    ) -> ExecutionRecord:
        """Fail a publication whose outcome is unknown; never retry the publication itself.

        Args:
            execution: Current execution record.
            now: Current database time.

        Returns:
            The failed record with a failure report queued.
        """
        return TransitionsBase._terminal_failure(
            execution,
            ExecutionError(
                code=ExecutionErrorCode.PROVIDER_FINISH_OUTCOME_UNKNOWN,
                message="provider finish outcome is unknown after work was interrupted",
            ),
            now,
            report=True,
        )

    @staticmethod
    def work_deadline_at(execution: ExecutionRecord) -> datetime:
        """Return the deadline of the execution's current work.

        Failure and cancellation reports use the terminal reporting deadline when supplied;
        other work uses the execution deadline.

        Args:
            execution: Current execution record.

        Returns:
            The deadline that bounds the current work.
        """
        if (
            execution.work_kind
            in {
                WorkKind.PROVIDER_FAILURE_REPORT,
                WorkKind.PROVIDER_CANCELLATION_REPORT,
            }
            and execution.terminal_report_deadline_at is not None
        ):
            return execution.terminal_report_deadline_at
        return execution.deadline_at

    @staticmethod
    def _require_kind(execution: ExecutionRecord, expected: WorkKind) -> None:
        """Reject settlement applied to a work kind different from the expected one.

        Args:
            execution: Current execution record.
            expected: Work kind the settlement applies to.

        Raises:
            StaleWorkClaimError: The execution's work kind differs.
        """
        if execution.work_kind is not expected:
            raise StaleWorkClaimError(f"claim is not {expected.value}")

    @staticmethod
    def _require_provider_kind(execution: ExecutionRecord) -> None:
        """Reject operations outside publication, failure reporting, or cancellation reporting.

        Args:
            execution: Current execution record.

        Raises:
            StaleWorkClaimError: The execution has no provider work.
        """
        if execution.work_kind not in {
            WorkKind.PROVIDER_FINISH,
            WorkKind.PROVIDER_FAILURE_REPORT,
            WorkKind.PROVIDER_CANCELLATION_REPORT,
        }:
            raise StaleWorkClaimError("claim is not provider work")

    def _retry(
        self,
        execution: ExecutionRecord,
        now: datetime,
        backoff: timedelta,
        *,
        status: ExecutionStatus,
    ) -> ExecutionRecord:
        """Release ownership and schedule a retry no later than the applicable work deadline.

        Args:
            execution: Current execution record.
            now: Current database time.
            backoff: Delay before the retry.
            status: Status to give the execution while it waits.

        Returns:
            The released record with its next attempt scheduled.
        """
        return replace(
            execution,
            status=status,
            next_attempt_at=min(now + backoff, self.work_deadline_at(execution)),
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
        """Record a safe failure, release the lease, and optionally queue a failure report.

        The report is new provider work, so it gets its own finish attempts.

        Args:
            execution: Current execution record.
            error: Safe error to record.
            now: Current database time.
            report: Whether to queue a failure report to the provider.

        Returns:
            The failed record.
        """
        return replace(
            execution,
            status=ExecutionStatus.FAILED,
            work_kind=(WorkKind.PROVIDER_FAILURE_REPORT if report else None),
            next_attempt_at=now if report else None,
            finish_attempts=0 if report else execution.finish_attempts,
            finish_mutation_started=False,
            lease_owner=None,
            fencing_token=None,
            lease_expires_at=None,
            error=error,
            updated_at=now,
        )

    @staticmethod
    def _clear_work(execution: ExecutionRecord, now: datetime) -> ExecutionRecord:
        """Remove pending work and ownership while preserving the already established public outcome.

        Args:
            execution: Current execution record.
            now: Current database time.

        Returns:
            The record with no work or lease.
        """
        return replace(
            execution,
            work_kind=None,
            next_attempt_at=None,
            lease_owner=None,
            fencing_token=None,
            lease_expires_at=None,
            updated_at=now,
        )
