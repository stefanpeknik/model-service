"""Claiming work and deciding lease renewal."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from model_service.scheduler.internal_state import (
    ExecutionRecord,
    RenewDirective,
    StaleWorkClaimError,
    WorkClaim,
    WorkKind,
)
from model_service.scheduler.status import ExecutionStatus


if TYPE_CHECKING:
    from datetime import datetime, timedelta

    from model_service.scheduler.transitions.expiry import ExpiryTransitions


from model_service.scheduler.transitions.common import (
    ClaimDecision,
    RenewalDecision,
    work_deadline_at,
)


class LeaseTransitions:
    """Claiming work and deciding lease renewal."""

    def __init__(self, expiry: ExpiryTransitions) -> None:
        """Keep the expiry rules a renewal past its deadline applies.

        Args:
            expiry: Deadline and lease-recovery rules.
        """
        self._expiry = expiry

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

        Processing and publication move the execution to running; terminal reports keep its status.

        Args:
            execution: Current execution record.
            owner: ID of the claiming worker.
            fencing_token: New token that identifies this attempt.
            lease_duration: How long the claim stays valid without renewal.
            now: Current database time.

        Returns:
            The leased record and the claim handed to the worker.

        Raises:
            StaleWorkClaimError: The execution has no due work.
        """
        work_kind = execution.work_kind
        if work_kind is None:
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
        return ClaimDecision(
            execution=claimed,
            claim=WorkClaim(
                execution_id=claimed.id,
                builder_id=claimed.builder_id,
                model_id=claimed.model_id,
                provider=claimed.identity.provider,
                kind=work_kind,
                invocation=claimed.invocation,
                protected_context=claimed.protected_context,
                attempt=claimed.current_attempt,
                fencing_token=fencing_token,
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
            execution: Current execution record.
            fencing_token: Token of the renewing attempt.
            lease_duration: Extension to apply from now.
            now: Current database time.

        Returns:
            The state update, if any, and the directive: CANCEL or LEASE_LOST for a
            lost lease, DEADLINE_EXCEEDED with the deadline transition, or CONTINUE with
            the extended lease.
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
        if work_deadline_at(execution) <= now:
            return RenewalDecision(
                execution=self._expiry.deadline(execution, now),
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
