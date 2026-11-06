"""Specify atomic persistence operations used by the scheduler and worker.

The PostgreSQL and in-memory adapters apply the same ExecutionTransitions rules.
Adapters translate storage failures into ExecutionRepositoryUnavailableError.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from collections.abc import Collection
    from datetime import timedelta

    from model_service.scheduler.contracts import Operation
    from model_service.scheduler.domain import (
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
        """Check that storage is accessible without requiring Ray.

        Raises:
            ExecutionRepositoryUnavailableError: Storage is unavailable.
        """
        ...

    async def maintain(self) -> None:
        """Expire deadlines, recover abandoned leases and delete expired inference results.

        Raises:
            ExecutionRepositoryUnavailableError: Storage is unavailable.
        """
        ...

    async def find_submission(
        self,
        identity: SubmissionIdentity,
    ) -> ExecutionRecord | None:
        """Find a provider-scoped replay key, returning None if absent and rejecting fingerprint conflicts.

        Args:
            identity: Replay identity of the submission.

        Returns:
            The existing execution, or None.

        Raises:
            ScheduleConflictError: The key was used for a request with a different fingerprint.
            ExecutionRepositoryUnavailableError: Storage is unavailable.
        """
        ...

    async def admit(self, admission: Admission) -> AdmissionResult:
        """Atomically insert a prepared execution (and its inference payload) or return the existing identical submission.

        Args:
            admission: Prepared execution to store.

        Returns:
            The stored execution and whether this call created it.

        Raises:
            ScheduleConflictError: A concurrent submission used the key for a different request.
            ExecutionRepositoryUnavailableError: Storage is unavailable.
        """
        ...

    async def get(self, execution_id: str) -> ExecutionRecord:
        """Return the internal record or raise UnknownExecutionError for an absent ID.

        Args:
            execution_id: ID of the execution.

        Returns:
            The record, without its protected context.

        Raises:
            UnknownExecutionError: No execution has this ID.
            ExecutionRepositoryUnavailableError: Storage is unavailable.
        """
        ...

    async def get_many(self, execution_ids: Collection[str]) -> list[ExecutionRecord]:
        """Return the records that exist, in the order requested.

        Args:
            execution_ids: IDs to read; unknown ones are skipped.

        Returns:
            The existing records in request order.

        Raises:
            ExecutionRepositoryUnavailableError: Storage is unavailable.
        """
        ...

    async def request_cancel(self, execution_id: str) -> ExecutionRecord:
        """Record cancellation and invalidate the active claim, leaving terminal executions unchanged.

        Args:
            execution_id: ID of the execution to cancel.

        Returns:
            The record after the request.

        Raises:
            UnknownExecutionError: No execution has this ID.
            ExecutionRepositoryUnavailableError: Storage is unavailable.
        """
        ...

    async def claim_due(
        self,
        owner: str,
        lease_duration: timedelta,
        operations: Collection[Operation] | None = None,
    ) -> WorkClaim | None:
        """Lease one eligible attempt of the given operations (all when None).

        Provider work comes first; processing respects each model's capacity for
        that operation. An inference claim carries its payload.

        Args:
            owner: ID of the claiming worker.
            lease_duration: How long the claim stays valid without renewal.
            operations: Operations to claim from, or None for all.

        Returns:
            The claim, or None when nothing is due.

        Raises:
            ExecutionRepositoryUnavailableError: Storage is unavailable.
        """
        ...

    async def claim_inference(
        self, owner: str, lease_duration: timedelta, limit: int
    ) -> list[WorkClaim]:
        """Lease up to `limit` due inference attempts, with payloads, at once.

        Args:
            owner: ID of the claiming worker.
            lease_duration: How long each claim stays valid without renewal.
            limit: Maximum number of claims.

        Returns:
            The claims; empty when nothing is due or limit is not positive.

        Raises:
            ExecutionRepositoryUnavailableError: Storage is unavailable.
        """
        ...

    async def inference_content(self, execution_id: str) -> bytes:
        """Return a succeeded inference execution's stored response bytes.

        Args:
            execution_id: ID of the inference execution.

        Returns:
            The response body.

        Raises:
            UnknownExecutionError: No execution has this ID.
            ResultUnavailableError: The execution has no stored response or it cannot be read.
            ExecutionRepositoryUnavailableError: Storage is unavailable.
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
            execution_id: ID of the claimed execution.
            fencing_token: Token of the renewing attempt.
            lease_duration: Extension to apply from now.

        Returns:
            The directive the worker must follow.

        Raises:
            UnknownExecutionError: No execution has this ID.
            ExecutionRepositoryUnavailableError: Storage is unavailable.
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
            execution_id: ID of the claimed execution.
            fencing_token: Token of the settling attempt.
            settlement: Outcome of the attempt.

        Returns:
            The record after the outcome.

        Raises:
            UnknownExecutionError: No execution has this ID.
            StaleWorkClaimError: The token or lease is no longer current, the deadline passed, or the work kind does not match.
            ExecutionRepositoryUnavailableError: Storage is unavailable.
        """
        ...

    async def mark_finish_mutation_started(
        self,
        execution_id: str,
        fencing_token: str,
    ) -> None:
        """Persist the mutation boundary before an external provider write so recovery knows its outcome may be uncertain.

        Args:
            execution_id: ID of the claimed execution.
            fencing_token: Token of the provider attempt.

        Raises:
            UnknownExecutionError: No execution has this ID.
            StaleWorkClaimError: The claim is not provider work or is no longer current.
            ExecutionRepositoryUnavailableError: Storage is unavailable.
        """
        ...


class ExecutionRepositoryUnavailableError(RuntimeError):
    """Hide driver-specific storage failures behind the repository's availability contract."""
