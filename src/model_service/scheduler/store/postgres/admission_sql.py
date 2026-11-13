"""Admission, lookup and cancellation of executions in PostgreSQL."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from model_service.scheduler.internal_state import (
    Admission,
    AdmissionResult,
    ExecutionRecord,
)
from model_service.scheduler.public import ScheduleConflictError
from model_service.scheduler.store.rows import ExecutionRow


if TYPE_CHECKING:
    from collections.abc import Collection

    from sqlalchemy.ext.asyncio import AsyncSession

    from model_service.scheduler.public import SubmissionIdentity
    from model_service.scheduler.store.postgres.base import PostgresDatabase


from model_service.scheduler.store.postgres.base import (
    _is_uuid,
)


class AdmissionOperations:
    """Admission, lookup and cancellation."""

    def __init__(self, db: PostgresDatabase) -> None:
        """Keep the database these operations run on.

        Args:
            db: Connection pool, transactions and row mapping.
        """
        self._db = db

    async def find_submission(
        self,
        identity: SubmissionIdentity,
    ) -> ExecutionRecord | None:
        """Read by provider and replay key, decrypt the record, and reject a different request fingerprint.

        Args:
            identity: Replay identity of the submission.

        Returns:
            The existing execution, or None.

        Raises:
            ScheduleConflictError: The key was used for a request with a different fingerprint.
            ExecutionRepositoryUnavailableError: The database is unavailable.
        """
        async with self._db.transaction() as session:
            row = await self._find_submission(session, identity)
            if row is None:
                return None
            execution = self._db.record(row)
            self._check_fingerprint(execution, identity)
            return execution

    async def admit(self, admission: Admission) -> AdmissionResult:
        """Encrypt protected context and atomically insert a queued record or return an identical concurrent replay.

        Args:
            admission: Prepared execution to store.

        Returns:
            The stored execution and whether this call created it.

        Raises:
            ScheduleConflictError: A concurrent submission used the key for a different request.
            ExecutionRepositoryUnavailableError: The database is unavailable.
        """
        return await self._insert(
            admission,
            str(uuid4()),
            self._db.cipher.encrypt(admission.protected_context),
        )

    async def _insert(
        self,
        admission: Admission,
        execution_id: str,
        encrypted_context: bytes,
    ) -> AdmissionResult:
        """Insert the execution row, or read the row a concurrent submission inserted first.

        Args:
            admission: Prepared execution to store.
            execution_id: ID for the new execution.
            encrypted_context: Encrypted provider context.

        Returns:
            The stored execution and whether this call created it.

        Raises:
            ScheduleConflictError: The existing row has a different fingerprint.
            RuntimeError: The conflicting row disappeared before it could be read.
            ExecutionRepositoryUnavailableError: The database is unavailable.
        """
        async with self._db.transaction() as session:
            execution = self._db.transitions.admission.admit(
                admission, execution_id, await self._db.database_now(session)
            )
            row = await session.scalar(
                insert(ExecutionRow)
                .values(
                    id=execution.id,
                    provider=execution.identity.provider,
                    submission_key=execution.identity.key,
                    request_fingerprint=execution.identity.fingerprint,
                    builder_id=execution.builder_id,
                    model_id=execution.model_id,
                    status=execution.status,
                    invocation=execution.invocation,
                    protected_context=encrypted_context,
                    deadline_at=execution.deadline_at,
                    terminal_report_deadline_at=execution.terminal_report_deadline_at,
                    work_kind=execution.work_kind,
                    next_attempt_at=execution.next_attempt_at,
                    processing_attempts=execution.processing_attempts,
                    processing_rejections=execution.processing_rejections,
                    finish_attempts=execution.finish_attempts,
                    finish_mutation_started=execution.finish_mutation_started,
                    priority=execution.priority,
                    created_at=execution.created_at,
                    updated_at=execution.updated_at,
                )
                .on_conflict_do_nothing(
                    index_elements=[ExecutionRow.provider, ExecutionRow.submission_key]
                )
                .returning(ExecutionRow)
            )
            if row is not None:
                return AdmissionResult(self._db.record(row), created=True)
            existing = await self._find_submission(session, admission.identity)
            if existing is None:
                raise RuntimeError("conflicting admission disappeared")
            execution = self._db.record(existing)
            self._check_fingerprint(execution, admission.identity)
            return AdmissionResult(execution, created=False)

    async def get(self, execution_id: str) -> ExecutionRecord:
        """Read and decode a domain record, raising UnknownExecutionError for an absent ID.

        Args:
            execution_id: ID of the execution.

        Returns:
            The record, without its protected context.

        Raises:
            UnknownExecutionError: No execution has this ID.
            ExecutionRepositoryUnavailableError: The database is unavailable.
        """
        async with self._db.transaction() as session:
            return self._db.record(await self._db.fetch(session, execution_id))

    async def get_many(self, execution_ids: Collection[str]) -> list[ExecutionRecord]:
        """Read existing records in the requested order, ignoring unknown IDs.

        Args:
            execution_ids: IDs to read; ones that are not UUIDs are skipped without a query.

        Returns:
            The existing records in request order.

        Raises:
            ExecutionRepositoryUnavailableError: The database is unavailable.
        """
        valid = [value for value in execution_ids if _is_uuid(value)]
        if not valid:
            return []
        async with self._db.transaction() as session:
            rows = await session.scalars(
                select(ExecutionRow).where(ExecutionRow.id.in_(valid))
            )
            found = {row.id: self._db.record(row) for row in rows}
        return [found[value] for value in valid if value in found]

    async def request_cancel(self, execution_id: str) -> ExecutionRecord:
        """Lock the execution row and persist cancellation with the old claim invalidated.

        Terminal outcomes remain unchanged.

        Args:
            execution_id: ID of the execution to cancel.

        Returns:
            The record after the request.

        Raises:
            UnknownExecutionError: No execution has this ID.
            ExecutionRepositoryUnavailableError: The database is unavailable.
        """
        async with self._db.transaction() as session:
            row = await self._db.fetch(session, execution_id, for_update=True)
            execution = self._db.record(row)
            cancelled = self._db.transitions.admission.cancel(
                execution, await self._db.database_now(session)
            )
            if cancelled is execution:
                return execution
            return await self._db.persist_state(session, row, cancelled)

    @staticmethod
    async def _find_submission(
        session: AsyncSession, identity: SubmissionIdentity
    ) -> ExecutionRow | None:
        """Query the unique provider/key identity in the caller's session without comparing its fingerprint.

        Args:
            session: Session of the caller's transaction.
            identity: Replay identity of the submission.

        Returns:
            The matching row, or None.
        """
        return await session.scalar(
            select(ExecutionRow).where(
                ExecutionRow.provider == identity.provider,
                ExecutionRow.submission_key == identity.key,
            )
        )

    @staticmethod
    def _check_fingerprint(
        execution: ExecutionRecord,
        identity: SubmissionIdentity,
    ) -> None:
        """Reject reuse of a provider/key identity with different request contents.

        Args:
            execution: Stored execution with that identity.
            identity: Identity of the new submission.

        Raises:
            ScheduleConflictError: The fingerprints differ.
        """
        if execution.identity.fingerprint != identity.fingerprint:
            raise ScheduleConflictError(
                "submission identity was reused with a different request"
            )
