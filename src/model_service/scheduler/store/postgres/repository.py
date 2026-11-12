"""The PostgreSQL ExecutionRepository, delegating to one operations object per concern."""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_service.scheduler.store.postgres.admission_sql import AdmissionOperations
from model_service.scheduler.store.postgres.attempt_outcome_sql import (
    SettlementOperations,
)
from model_service.scheduler.store.postgres.base import PostgresDatabase
from model_service.scheduler.store.postgres.claim_sql import ClaimOperations
from model_service.scheduler.store.postgres.maintenance_sql import (
    MaintenanceOperations,
)
from model_service.scheduler.store.repository import ExecutionRepository
from model_service.scheduler.store.rows import DEFAULT_SCHEMA


if TYPE_CHECKING:
    from collections.abc import Collection
    from datetime import timedelta

    from model_service.scheduler.internal_state import (
        Admission,
        AdmissionResult,
        ExecutionRecord,
        RenewDirective,
        RetryPolicy,
        Settlement,
        WorkClaim,
    )
    from model_service.scheduler.public import SubmissionIdentity
    from model_service.scheduler.store.protected_context import ProtectedContextCipher


class PostgresExecutionRepository(ExecutionRepository):
    """Provide durable replay-safe admission, leased claims, and recovery using the shared state machine.

    Each operation is delegated to the operations object of its concern:
    admission, claims, settlement or maintenance.
    """

    def __init__(
        self,
        dsn: str,
        *,
        cipher: ProtectedContextCipher,
        schema: str = DEFAULT_SCHEMA,
        per_model_limit: int = 1,
        processing_policy: RetryPolicy | None = None,
        finish_policy: RetryPolicy | None = None,
        pool_size: int = 10,
    ) -> None:
        """Configure the database and the operations built over it; no connection opens here.

        Args:
            dsn: libpq DSN or PostgreSQL URL.
            cipher: Cipher for the stored provider context.
            schema: PostgreSQL schema holding the execution table.
            per_model_limit: Concurrent builder executions allowed per model.
            processing_policy: Processing retry policy; defaults to 3 attempts, 5 s to 1 min backoff.
            finish_policy: Provider finish retry policy; same default.
            pool_size: Pooled connections, with as many again allowed as overflow.

        Raises:
            ValueError: schema is not an identifier or the per-model limit is not positive.
        """
        self._db = PostgresDatabase(
            dsn,
            cipher=cipher,
            schema=schema,
            per_model_limit=per_model_limit,
            processing_policy=processing_policy,
            finish_policy=finish_policy,
            pool_size=pool_size,
        )
        self._admission = AdmissionOperations(self._db)
        self._claims = ClaimOperations(self._db)
        self._settlement = SettlementOperations(self._db)
        self._maintenance = MaintenanceOperations(self._db)

    async def aclose(self) -> None:
        """Dispose of the connection pool."""
        await self._db.aclose()

    async def initialize(self) -> None:
        """Apply pending Yoyo migrations under a schema-specific initialization lock."""
        await self._db.initialize()

    async def check_ready(self) -> None:
        """Query the execution table to confirm database connectivity and schema availability.

        Raises:
            ExecutionRepositoryUnavailableError: The database is unavailable.
        """
        await self._db.check_ready()

    async def maintain(self) -> None:
        """Expire deadlines and recover abandoned claims.

        Raises:
            ExecutionRepositoryUnavailableError: The database is unavailable.
        """
        await self._maintenance.maintain()

    async def find_submission(
        self,
        identity: SubmissionIdentity,
    ) -> ExecutionRecord | None:
        """Return the execution admitted under a replay identity, if any.

        Args:
            identity: Replay identity of the submission.

        Returns:
            The execution, or None if none was admitted under the identity.

        Raises:
            SubmissionConflictError: The identity was admitted with a different request.
            ExecutionRepositoryUnavailableError: The database is unavailable.
        """
        return await self._admission.find_submission(identity)

    async def admit(self, admission: Admission) -> AdmissionResult:
        """Admit a submission, or return the execution already admitted under its identity.

        Args:
            admission: Validated submission to admit.

        Returns:
            The execution and whether this call created it.

        Raises:
            SubmissionConflictError: The identity was admitted with a different request.
            ExecutionRepositoryUnavailableError: The database is unavailable.
        """
        return await self._admission.admit(admission)

    async def get(self, execution_id: str) -> ExecutionRecord:
        """Return one execution.

        Args:
            execution_id: ID of the execution.

        Returns:
            The execution record.

        Raises:
            UnknownExecutionError: No execution has this ID.
            ExecutionRepositoryUnavailableError: The database is unavailable.
        """
        return await self._admission.get(execution_id)

    async def get_many(self, execution_ids: Collection[str]) -> list[ExecutionRecord]:
        """Return the executions with the given IDs.

        Args:
            execution_ids: IDs of the executions.

        Returns:
            The records found, in the order of the IDs.

        Raises:
            ExecutionRepositoryUnavailableError: The database is unavailable.
        """
        return await self._admission.get_many(execution_ids)

    async def request_cancel(self, execution_id: str) -> ExecutionRecord:
        """Cancel a nonterminal execution and queue its cancellation report.

        Args:
            execution_id: ID of the execution.

        Returns:
            The cancelled record, or the record unchanged if it already ended.

        Raises:
            UnknownExecutionError: No execution has this ID.
            ExecutionRepositoryUnavailableError: The database is unavailable.
        """
        return await self._admission.request_cancel(execution_id)

    async def claim_due(
        self,
        owner: str,
        lease_duration: timedelta,
    ) -> WorkClaim | None:
        """Lease the next due attempt, if any.

        Args:
            owner: ID of the claiming worker.
            lease_duration: How long the claim stays valid without renewal.

        Returns:
            The claim, or None if no work is due.

        Raises:
            ExecutionRepositoryUnavailableError: The database is unavailable.
        """
        return await self._claims.claim_due(owner, lease_duration)

    async def renew(
        self,
        execution_id: str,
        fencing_token: str,
        lease_duration: timedelta,
    ) -> RenewDirective:
        """Extend a live lease, or tell the worker why it must stop.

        Args:
            execution_id: ID of the claimed execution.
            fencing_token: Token of the renewing attempt.
            lease_duration: Extension to apply from now.

        Returns:
            The directive the worker must follow.

        Raises:
            UnknownExecutionError: No execution has this ID.
            ExecutionRepositoryUnavailableError: The database is unavailable.
        """
        return await self._claims.renew(execution_id, fencing_token, lease_duration)

    async def settle(
        self,
        execution_id: str,
        fencing_token: str,
        settlement: Settlement,
    ) -> ExecutionRecord:
        """Record the outcome of a claimed attempt.

        Args:
            execution_id: ID of the claimed execution.
            fencing_token: Token of the settling attempt.
            settlement: Outcome of the attempt.

        Returns:
            The record after the outcome.

        Raises:
            StaleWorkClaimError: The claim is no longer current or its deadline passed.
            UnknownExecutionError: No execution has this ID.
            ExecutionRepositoryUnavailableError: The database is unavailable.
        """
        return await self._settlement.settle(execution_id, fencing_token, settlement)

    async def mark_finish_mutation_started(
        self,
        execution_id: str,
        fencing_token: str,
    ) -> None:
        """Mark that a live provider attempt is about to write to the provider.

        Args:
            execution_id: ID of the claimed execution.
            fencing_token: Token of the provider attempt.

        Raises:
            StaleWorkClaimError: The claim is no longer current.
            UnknownExecutionError: No execution has this ID.
            ExecutionRepositoryUnavailableError: The database is unavailable.
        """
        await self._settlement.mark_finish_mutation_started(execution_id, fencing_token)
