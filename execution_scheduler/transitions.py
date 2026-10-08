"""Compute execution state changes without database or network I/O.

Both persistence adapters use these immutable decisions for admission, leases,
deadlines, retries, and provider finish. Repositories supply time and make each decision
atomic.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum
from typing import TYPE_CHECKING

from execution_scheduler.contracts import (
    ExecutionError,
    ExecutionErrorCode,
    ExecutionStatus,
)
from execution_scheduler.domain import (
    Admission,
    ExecutionRecord,
    ProcessingFailed,
    ProcessingRejected,
    ProcessingSucceeded,
    ProviderFinishFailed,
    ProviderFinishSucceeded,
    RenewDirective,
    StaleWorkClaimError,
    WorkClaim,
    WorkKind,
)


if TYPE_CHECKING:
    from datetime import datetime, timedelta

    from execution_scheduler.domain import RetryPolicy, Settlement


@dataclass(frozen=True, slots=True)
class ClaimDecision:
    """Pair the new leased record with the attempt description returned to its worker.

    Attributes:
        execution: Immutable execution record after this state decision.
        claim: Leased worker attempt corresponding to the updated execution.
    """

    execution: ExecutionRecord
    claim: WorkClaim


@dataclass(frozen=True, slots=True)
class RenewalDecision:
    """Return an optional state update and the instruction a lease-renewing worker must follow.

    Attributes:
        execution: Immutable execution record after this state decision.
        directive: Whether the worker may continue or must stop.
    """

    execution: ExecutionRecord | None
    directive: RenewDirective


class SettlementRejection(StrEnum):
    """Explain why an attempt's outcome cannot be accepted into durable state."""

    STALE_CLAIM = "work claim is no longer current"
    DEADLINE_EXCEEDED = "execution deadline was exceeded"


@dataclass(frozen=True, slots=True)
class SettlementDecision:
    """Carry an accepted state update or a rejection, including deadline changes that must still be persisted.

    Attributes:
        execution: Immutable execution record after this state decision.
        rejection: Reason settlement is refused, or None for acceptance.
    """

    execution: ExecutionRecord | None
    rejection: SettlementRejection | None = None

    def accepted_execution(self) -> ExecutionRecord:
        """Return the accepted record or raise StaleWorkClaimError after any required state update is persisted.

        Returns:
            ExecutionRecord: Internal durable execution snapshot, including work and
                ownership fields.

        Raises:
            StaleWorkClaimError: The token, lease, deadline, or work kind no longer
                permits this operation.
        """
        if self.rejection is not None:
            raise StaleWorkClaimError(self.rejection.value)
        assert self.execution is not None
        return self.execution


class ExecutionTransitions:
    """Apply the shared state machine with separate processing and provider retry policies."""

    def __init__(
        self,
        processing_policy: RetryPolicy,
        finish_policy: RetryPolicy,
    ) -> None:
        """Retain the policies used to bound attempts and calculate retry delays.

        Args:
            processing_policy: Attempt/backoff limits for workflow computation.
            finish_policy: Attempt/backoff limits for publication and terminal
                reporting.
        """
        self._processing_policy = processing_policy
        self._finish_policy = finish_policy

    @staticmethod
    def admit(
        admission: Admission,
        execution_id: str,
        now: datetime,
    ) -> ExecutionRecord:
        """Create a queued execution with immediate processing due, zero attempts, and no lease or outcome.

        Args:
            admission: Prepared submission containing replay identity, invocation,
                deadlines, and private context.
            execution_id: Durable execution identifier, not an external provider job ID.
            now: Aware current time supplied by the repository or validation clock.

        Returns:
            ExecutionRecord: Queued, unleased record with processing immediately due and
                zero consumed attempts.
        """
        return ExecutionRecord(
            id=execution_id,
            identity=admission.identity,
            workflow_id=admission.workflow_id,
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
        )

    @staticmethod
    def cancel(execution: ExecutionRecord, now: datetime) -> ExecutionRecord:
        """Make a nonterminal execution cancelled, fence its attempt, and queue provider cancellation reporting.

        Args:
            execution: Current immutable execution record; transitions return a
                replacement without mutating it.
            now: Aware current time supplied by the repository or validation clock.

        Returns:
            ExecutionRecord: Cancelled and fenced record with provider cancellation
                reporting due, or the original terminal record unchanged.
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
            updated_at=now,
        )

    @staticmethod
    def claim(
        execution: ExecutionRecord,
        *,
        owner: str,
        fencing_token: str,
        lease_duration: timedelta,
        now: datetime,
    ) -> ClaimDecision:
        """Assign ownership and a fencing token, advance the appropriate attempt counter, and return the leased record and claim.

        Args:
            execution: Current immutable execution record; transitions return a
                replacement without mutating it.
            owner: Unique worker identity recorded on the durable lease.
            fencing_token: Current claim token; stale tokens cannot mutate execution
                state.
            lease_duration: How long a claimed attempt retains ownership without
                renewal.
            now: Aware current time supplied by the repository or validation clock.

        Returns:
            ClaimDecision: New leased execution state and the corresponding worker
                claim.

        Raises:
            StaleWorkClaimError: The token, lease, deadline, or work kind no longer
                permits this operation.
        """
        if execution.work_kind is None:
            raise StaleWorkClaimError("execution has no due work")
        processing_attempts = execution.processing_attempts
        finish_attempts = execution.finish_attempts
        if execution.work_kind is WorkKind.PROCESSING:
            processing_attempts += 1
        else:
            finish_attempts += 1
        claimed = replace(
            execution,
            status=(
                ExecutionStatus.RUNNING
                if execution.work_kind
                in {WorkKind.PROCESSING, WorkKind.PROVIDER_FINISH}
                else execution.status
            ),
            processing_attempts=processing_attempts,
            finish_attempts=finish_attempts,
            lease_owner=owner,
            fencing_token=fencing_token,
            lease_expires_at=now + lease_duration,
            updated_at=now,
        )
        assert claimed.work_kind is not None
        assert claimed.fencing_token is not None
        attempt = (
            claimed.processing_attempts
            if claimed.work_kind is WorkKind.PROCESSING
            else claimed.finish_attempts
        )
        return ClaimDecision(
            execution=claimed,
            claim=WorkClaim(
                execution_id=claimed.id,
                workflow_id=claimed.workflow_id,
                provider=claimed.identity.provider,
                kind=claimed.work_kind,
                invocation=claimed.invocation,
                protected_context=claimed.protected_context,
                attempt=attempt,
                fencing_token=claimed.fencing_token,
                result=claimed.result,
                error=claimed.error,
            ),
        )

    def decide_renewal(
        self,
        execution: ExecutionRecord,
        *,
        fencing_token: str,
        lease_duration: timedelta,
        now: datetime,
    ) -> RenewalDecision:
        """Reject a stale or expired lease, apply an elapsed work deadline, or extend valid ownership.

        Args:
            execution: Current immutable execution record; transitions return a
                replacement without mutating it.
            fencing_token: Current claim token; stale tokens cannot mutate execution
                state.
            lease_duration: How long a claimed attempt retains ownership without
                renewal.
            now: Aware current time supplied by the repository or validation clock.

        Returns:
            RenewalDecision: Optional durable update and the worker's ownership/deadline
                directive.
        """
        if (
            execution.fencing_token != fencing_token
            or execution.lease_expires_at is None
            or execution.lease_expires_at <= now
        ):
            return RenewalDecision(
                execution=None,
                directive=(
                    RenewDirective.CANCEL
                    if execution.status is ExecutionStatus.CANCELLED
                    else RenewDirective.LEASE_LOST
                ),
            )
        if self.work_deadline_at(execution) <= now:
            return RenewalDecision(
                execution=self.deadline(execution, now),
                directive=RenewDirective.DEADLINE_EXCEEDED,
            )
        return RenewalDecision(
            execution=replace(
                execution,
                lease_expires_at=now + lease_duration,
                updated_at=now,
            ),
            directive=RenewDirective.CONTINUE,
        )

    def decide_settlement(
        self,
        execution: ExecutionRecord,
        fencing_token: str,
        settlement: Settlement,
        now: datetime,
    ) -> SettlementDecision:
        """Check token, deadline, and lease before applying an outcome; preserve a deadline update even when settlement is rejected.

        Args:
            execution: Current immutable execution record; transitions return a
                replacement without mutating it.
            fencing_token: Current claim token; stale tokens cannot mutate execution
                state.
            settlement: Processing or provider outcome produced by the current leased
                attempt.
            now: Aware current time supplied by the repository or validation clock.

        Returns:
            SettlementDecision: Accepted state update or rejection, possibly with a
                deadline update to persist.
        """
        if execution.fencing_token != fencing_token:
            return SettlementDecision(
                execution=None,
                rejection=SettlementRejection.STALE_CLAIM,
            )
        if self.work_deadline_at(execution) <= now:
            return SettlementDecision(
                execution=self.deadline(execution, now),
                rejection=SettlementRejection.DEADLINE_EXCEEDED,
            )
        if execution.lease_expires_at is None or execution.lease_expires_at <= now:
            return SettlementDecision(
                execution=None,
                rejection=SettlementRejection.STALE_CLAIM,
            )
        return SettlementDecision(self.settle(execution, settlement, now))

    def mark_finish_mutation_started(
        self,
        execution: ExecutionRecord,
        fencing_token: str,
        now: datetime,
    ) -> ExecutionRecord:
        """Validate a live provider claim and mark that an external write is about to begin.

        Args:
            execution: Current immutable execution record; transitions return a
                replacement without mutating it.
            fencing_token: Current claim token; stale tokens cannot mutate execution
                state.
            now: Aware current time supplied by the repository or validation clock.

        Returns:
            ExecutionRecord: Current provider record with finish_mutation_started set
                and updated_at advanced.

        Raises:
            StaleWorkClaimError: The token, lease, deadline, or work kind no longer
                permits this operation.
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

        Processing success stores the result and queues finish rather than marking
        success. Confirmed rejection restores the processing attempt budget. Unknown
        provider mutation outcomes are never blindly retried.

        Args:
            execution: Current immutable execution record; transitions return a
                replacement without mutating it.
            settlement: Processing or provider outcome produced by the current leased
                attempt.
            now: Aware current time supplied by the repository or validation clock.

        Returns:
            ExecutionRecord: Next processing/provider state, with ownership released and
                retries or terminal reporting scheduled as required.

        Raises:
            StaleWorkClaimError: The settlement belongs to a different processing or
                provider work kind.
            TypeError: settlement is not one of the supported settlement types.
        """
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
                return self._terminal_failure(execution, settlement.error, now)
            return self._clear_work(execution, now)
        raise TypeError("unknown work settlement")

    def recover_expired(
        self,
        execution: ExecutionRecord,
        now: datetime,
    ) -> ExecutionRecord:
        """Recover abandoned work through retry or terminal failure, respecting attempt limits and uncertain provider writes.

        Args:
            execution: Current immutable execution record; transitions return a
                replacement without mutating it.
            now: Aware current time supplied by the repository or validation clock.

        Returns:
            ExecutionRecord: Retried or terminal record after abandonment, preserving
                stored results and uncertainty at the provider mutation boundary.
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
        """Fail overdue active work or stop overdue terminal reporting; interrupted publication can instead become outcome-unknown.

        Args:
            execution: Current immutable execution record; transitions return a
                replacement without mutating it.
            now: Aware current time supplied by the repository or validation clock.

        Returns:
            ExecutionRecord: Deadline failure or cleared terminal reporting work,
                possibly unchanged while a later report deadline remains open.
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

    @staticmethod
    def _finish_outcome_unknown(
        execution: ExecutionRecord,
        now: datetime,
    ) -> ExecutionRecord:
        """Fail result publication whose external outcome cannot be established, without scheduling another write.

        Args:
            execution: Current immutable execution record; transitions return a
                replacement without mutating it.
            now: Aware current time supplied by the repository or validation clock.

        Returns:
            ExecutionRecord: Terminal provider_finish_outcome_unknown failure with no
                further publication due.
        """
        return ExecutionTransitions._terminal_failure(
            execution,
            ExecutionError(
                code=ExecutionErrorCode.PROVIDER_FINISH_OUTCOME_UNKNOWN,
                message="provider finish outcome is unknown after work was interrupted",
            ),
            now,
        )

    @staticmethod
    def work_deadline_at(execution: ExecutionRecord) -> datetime:
        """Use the terminal reporting deadline for failure/cancellation reports when supplied; otherwise use the execution deadline.

        Args:
            execution: Current immutable execution record; transitions return a
                replacement without mutating it.

        Returns:
            datetime: Execution deadline, or the provider reporting deadline for
                terminal report work when configured.
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
        """Reject settlement applied to a work kind different from the expected processing or provider phase.

        Args:
            execution: Current immutable execution record; transitions return a
                replacement without mutating it.
            expected: Required status, work kind, or permitted lifecycle states for this
                check.

        Returns:
            None: No return value.

        Raises:
            StaleWorkClaimError: The token, lease, deadline, or work kind no longer
                permits this operation.
        """
        if execution.work_kind is not expected:
            raise StaleWorkClaimError(f"claim is not {expected.value}")

    @staticmethod
    def _require_provider_kind(execution: ExecutionRecord) -> None:
        """Reject operations outside publication, failure reporting, or cancellation reporting.

        Args:
            execution: Current immutable execution record; transitions return a
                replacement without mutating it.

        Returns:
            None: No return value.

        Raises:
            StaleWorkClaimError: The token, lease, deadline, or work kind no longer
                permits this operation.
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
            execution: Current immutable execution record; transitions return a
                replacement without mutating it.
            now: Aware current time supplied by the repository or validation clock.
            backoff: Delay before this work becomes eligible for another attempt.
            status: Expected caller-visible execution status.

        Returns:
            ExecutionRecord: Unleased record with the requested status and a retry due
                no later than its work deadline.
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
        """Record a safe failure, release the lease, and optionally queue provider failure reporting.

        Args:
            execution: Current immutable execution record; transitions return a
                replacement without mutating it.
            error: Safe domain error or injected integration exception to classify.
            now: Aware current time supplied by the repository or validation clock.
            report: Whether terminal failure should leave provider failure-report work
                due.

        Returns:
            ExecutionRecord: Failed, unleased record, optionally retaining due provider
                failure-report work.
        """
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
        """Remove pending work and ownership while preserving the already established public outcome.

        Args:
            execution: Current immutable execution record; transitions return a
                replacement without mutating it.
            now: Aware current time supplied by the repository or validation clock.

        Returns:
            ExecutionRecord: Record retaining its public outcome but with pending work
                and ownership cleared.
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
