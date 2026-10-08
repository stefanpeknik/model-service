"""Specify atomic persistence operations used by the scheduler and worker.

The PostgreSQL and in-memory adapters apply the same ExecutionTransitions rules.
Adapters translate storage failures into ExecutionRepositoryUnavailableError.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from datetime import timedelta

    from execution_scheduler.domain import (
        Admission,
        AdmissionResult,
        ExecutionRecord,
        RenewDirective,
        Settlement,
        SubmissionIdentity,
        WorkClaim,
    )


class ExecutionRepository(Protocol):
    """Provide replay-safe admission, leased claims, settlement, and maintenance of durable executions."""

    async def check_ready(self) -> None:
        """Check that storage is accessible without requiring the workflow compute service.

        Returns:
            None: No return value.

        Raises:
            ExecutionRepositoryUnavailableError: Database access or the storage
                transaction fails.
        """
        ...

    async def maintain(self) -> None:
        """Expire execution deadlines and recover abandoned leases independently of worker dispatch.

        Returns:
            None: No return value.

        Raises:
            ExecutionRepositoryUnavailableError: Database access or the storage
                transaction fails.
        """
        ...

    async def find_submission(
        self,
        identity: SubmissionIdentity,
    ) -> ExecutionRecord | None:
        """Find a provider-scoped replay key, returning None if absent and rejecting fingerprint conflicts.

        Args:
            identity: Provider-scoped replay key and request fingerprint.

        Returns:
            ExecutionRecord | None: Matching internal record, or None if the
                provider/key pair has never been admitted.

        Raises:
            ExecutionRepositoryUnavailableError: Database access or the storage
                transaction fails.
            ScheduleConflictError: A provider submission key already exists with
                different request contents.
        """
        ...

    async def admit(self, admission: Admission) -> AdmissionResult:
        """Atomically insert a prepared execution or return the existing identical submission.

        Concurrent requests must converge on one record; a different fingerprint raises
        ScheduleConflictError.

        Args:
            admission: Prepared submission containing replay identity, invocation,
                deadlines, and private context.

        Returns:
            AdmissionResult: The admitted execution and whether this operation created
                it.

        Raises:
            ExecutionRepositoryUnavailableError: Database access or the storage
                transaction fails.
            ScheduleConflictError: A provider submission key already exists with
                different request contents.
        """
        ...

    async def get(self, execution_id: str) -> ExecutionRecord:
        """Return the internal record or raise UnknownExecutionError for an absent ID.

        Args:
            execution_id: Durable execution identifier, not an external provider job ID.

        Returns:
            ExecutionRecord: Internal durable execution snapshot, including work and
                ownership fields.

        Raises:
            ExecutionRepositoryUnavailableError: Database access or the storage
                transaction fails.
            UnknownExecutionError: The execution ID is absent from storage.
        """
        ...

    async def request_cancel(self, execution_id: str) -> ExecutionRecord:
        """Record cancellation and invalidate the active claim, leaving terminal executions unchanged.

        Args:
            execution_id: Durable execution identifier, not an external provider job ID.

        Returns:
            ExecutionRecord: Internal durable execution snapshot, including work and
                ownership fields.

        Raises:
            ExecutionRepositoryUnavailableError: Database access or the storage
                transaction fails.
            UnknownExecutionError: The execution ID is absent from storage.
        """
        ...

    async def claim_due(
        self,
        owner: str,
        lease_duration: timedelta,
    ) -> WorkClaim | None:
        """Lease one eligible attempt, prioritizing provider work and enforcing processing capacity per workflow.

        Args:
            owner: Unique worker identity recorded on the durable lease.
            lease_duration: How long a claimed attempt retains ownership without
                renewal.

        Returns:
            WorkClaim | None: One eligible leased attempt, or None if no due work can be
                claimed within capacity.

        Raises:
            ExecutionRepositoryUnavailableError: Database access or the storage
                transaction fails.
        """
        ...

    async def renew(
        self,
        execution_id: str,
        fencing_token: str,
        lease_duration: timedelta,
    ) -> RenewDirective:
        """Extend a current lease or return a stop directive when ownership or the work deadline is lost.

        Args:
            execution_id: Durable execution identifier, not an external provider job ID.
            fencing_token: Current claim token; stale tokens cannot mutate execution
                state.
            lease_duration: How long a claimed attempt retains ownership without
                renewal.

        Returns:
            RenewDirective: Instruction to continue or stop the current attempt.

        Raises:
            ExecutionRepositoryUnavailableError: Database access or the storage
                transaction fails.
            UnknownExecutionError: The execution ID is absent from storage.
        """
        ...

    async def settle(
        self,
        execution_id: str,
        fencing_token: str,
        settlement: Settlement,
    ) -> ExecutionRecord:
        """Apply an attempt's outcome only with a current token, unexpired lease, and valid deadline.

        Args:
            execution_id: Durable execution identifier, not an external provider job ID.
            fencing_token: Current claim token; stale tokens cannot mutate execution
                state.
            settlement: Processing or provider outcome produced by the current leased
                attempt.

        Returns:
            ExecutionRecord: Internal durable execution snapshot, including work and
                ownership fields.

        Raises:
            ExecutionRepositoryUnavailableError: Database access or the storage
                transaction fails.
            StaleWorkClaimError: The token, lease, deadline, or work kind no longer
                permits this operation.
            UnknownExecutionError: The execution ID is absent from storage.
        """
        ...

    async def mark_finish_mutation_started(
        self,
        execution_id: str,
        fencing_token: str,
    ) -> None:
        """Persist the mutation boundary before an external provider write so recovery knows its outcome may be uncertain.

        Args:
            execution_id: Durable execution identifier, not an external provider job ID.
            fencing_token: Current claim token; stale tokens cannot mutate execution
                state.

        Returns:
            None: No return value.

        Raises:
            ExecutionRepositoryUnavailableError: Database access or the storage
                transaction fails.
            StaleWorkClaimError: The token, lease, deadline, or work kind no longer
                permits this operation.
            UnknownExecutionError: The execution ID is absent from storage.
        """
        ...


class ExecutionRepositoryUnavailableError(RuntimeError):
    """Hide driver-specific storage failures behind the repository's availability contract."""
