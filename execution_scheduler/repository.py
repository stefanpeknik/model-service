"""Specify atomic persistence operations used by the scheduler and worker.

The PostgreSQL and in-memory adapters apply the same ExecutionTransitions rules.
Adapters translate storage failures into ExecutionRepositoryUnavailableError.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from collections.abc import Collection
    from datetime import timedelta

    from execution_scheduler.contracts import Operation
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
        """Check that storage is accessible without requiring the workflow compute service."""
        ...

    async def maintain(self) -> None:
        """Expire deadlines, recover abandoned leases and delete expired inference results."""
        ...

    async def find_submission(
        self,
        identity: SubmissionIdentity,
    ) -> ExecutionRecord | None:
        """Find a provider-scoped replay key, returning None if absent and rejecting fingerprint conflicts."""
        ...

    async def admit(self, admission: Admission) -> AdmissionResult:
        """Atomically insert a prepared execution (and its inference payload) or return the existing identical submission."""
        ...

    async def get(self, execution_id: str) -> ExecutionRecord:
        """Return the internal record or raise UnknownExecutionError for an absent ID."""
        ...

    async def get_many(self, execution_ids: Collection[str]) -> list[ExecutionRecord]:
        """Return the records that exist, in the order requested."""
        ...

    async def request_cancel(self, execution_id: str) -> ExecutionRecord:
        """Record cancellation and invalidate the active claim, leaving terminal executions unchanged."""
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
        """
        ...

    async def inference_content(self, execution_id: str) -> bytes:
        """Return a succeeded inference execution's stored response bytes.

        Raises UnknownExecutionError, or ResultUnavailableError when there is none.
        """
        ...

    async def renew(
        self,
        execution_id: str,
        fencing_token: str,
        lease_duration: timedelta,
    ) -> RenewDirective:
        """Extend a current lease or return a stop directive when ownership or the work deadline is lost."""
        ...

    async def settle(
        self,
        execution_id: str,
        fencing_token: str,
        settlement: Settlement,
    ) -> ExecutionRecord:
        """Apply an attempt's outcome only with a current token, unexpired lease, and valid deadline."""
        ...

    async def mark_finish_mutation_started(
        self,
        execution_id: str,
        fencing_token: str,
    ) -> None:
        """Persist the mutation boundary before an external provider write so recovery knows its outcome may be uncertain."""
        ...


class ExecutionRepositoryUnavailableError(RuntimeError):
    """Hide driver-specific storage failures behind the repository's availability contract."""
