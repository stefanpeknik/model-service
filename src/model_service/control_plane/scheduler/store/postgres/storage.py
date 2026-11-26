"""Executions in PostgreSQL: rows read and written in pooled transactions.

Row locks guard each record's changes and one transaction-scoped advisory lock
serializes claims. Database time governs leases and deadlines, so every
control-plane replica shares one clock.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING
from uuid import UUID

from psycopg import AsyncConnection
from psycopg import Error as PsycopgError
from sqlalchemy import func, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from model_service.control_plane.scheduler.internal_state import ExecutionRecord
from model_service.control_plane.scheduler.public import (
    ExecutionError,
    SubmissionIdentity,
    UnknownExecutionError,
)
from model_service.control_plane.scheduler.store.migrate import (
    initialize_execution_schema,
)
from model_service.control_plane.scheduler.store.rows import (
    DEFAULT_SCHEMA,
    ExecutionRow,
)
from model_service.control_plane.scheduler.store.storage import (
    ExecutionRepositoryUnavailableError,
    ExecutionStorage,
    StorageUnit,
)
from model_service.control_plane.scheduler.targets import ExecutionTarget


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Collection
    from datetime import datetime


# The key of the advisory lock that serializes claims across replicas.
CLAIM_LOCK = "model_service:claims"


class PostgresStorage(ExecutionStorage):
    """A pooled connection to the execution table in one schema."""

    def __init__(
        self, dsn: str, *, schema: str = DEFAULT_SCHEMA, pool_size: int = 10
    ) -> None:
        """Configure a pooled async engine; no connection opens here.

        Args:
            dsn: libpq DSN or PostgreSQL URL.
            schema: PostgreSQL schema holding the execution table.
            pool_size: Pooled connections, with as many again allowed as overflow.

        Raises:
            ValueError: schema is not an identifier.
        """
        if not schema or not schema.replace("_", "a").isalnum():
            raise ValueError("PostgreSQL schema must be an identifier")
        self._schema = schema
        self._dsn = dsn

        async def connect() -> AsyncConnection:
            """Open a psycopg connection from either a libpq DSN or PostgreSQL URL."""
            return await AsyncConnection.connect(dsn)

        # A new connection per transaction (fork plus SCRAM authentication) made
        # PostgreSQL a bottleneck.
        self._engine = create_async_engine(
            "postgresql+psycopg://",
            async_creator=connect,
            pool_size=pool_size,
            max_overflow=pool_size,
            pool_pre_ping=True,
            pool_recycle=300,
            hide_parameters=True,
            execution_options={"schema_translate_map": {DEFAULT_SCHEMA: schema}},
        )
        self._sessions = async_sessionmaker(self._engine, expire_on_commit=False)

    async def initialize(self) -> None:
        """Apply pending Yoyo migrations under a schema-specific initialization lock."""
        await initialize_execution_schema(self._dsn, schema=self._schema)

    async def aclose(self) -> None:
        """Dispose of the connection pool."""
        await self._engine.dispose()

    async def check_ready(self) -> None:
        """Query the execution table to confirm connectivity and the schema.

        Raises:
            ExecutionRepositoryUnavailableError: The database is unavailable.
        """
        async with self._transaction() as session:
            await session.execute(select(ExecutionRow.id).limit(1))

    @asynccontextmanager
    async def unit(self) -> AsyncIterator[StorageUnit]:
        """One transaction, committed on success; database errors become unavailability.

        Yields:
            The unit over the open transaction.

        Raises:
            ExecutionRepositoryUnavailableError: The database is unavailable.
        """
        async with self._transaction() as session:
            yield _PostgresUnit(session)

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[AsyncSession]:
        """One session and transaction, committed on success.

        Yields:
            The session of the open transaction.

        Raises:
            ExecutionRepositoryUnavailableError: The database is unavailable.
        """
        try:
            async with self._sessions.begin() as session:
                yield session
        except (SQLAlchemyError, PsycopgError) as error:
            raise ExecutionRepositoryUnavailableError(
                "execution repository is unavailable"
            ) from error


class _PostgresUnit(StorageUnit):
    """Reads and writes rows in the caller's transaction."""

    def __init__(self, session: AsyncSession) -> None:
        """Keep the session of the caller's transaction.

        Args:
            session: The open session.
        """
        self._session = session

    async def now(self) -> datetime:
        """Read clock_timestamp, the clock every replica shares."""
        return (
            await self._session.execute(select(func.clock_timestamp()))
        ).scalar_one()

    async def lock_claims(self) -> None:
        """Take the transaction-scoped claim lock, waiting for another claim to end."""
        await self._session.execute(
            select(func.pg_advisory_xact_lock(func.hashtextextended(CLAIM_LOCK, 0)))
        )

    async def insert(self, execution: ExecutionRecord, sealed_context: bytes) -> bool:
        """Insert a row unless the provider and key are taken; a racing insert waits for the other.

        Args:
            execution: The new record.
            sealed_context: Its encrypted provider context.

        Returns:
            False when an execution with the same provider and key exists.
        """
        inserted = await self._session.scalar(
            insert(ExecutionRow)
            .values(
                id=execution.id,
                provider=execution.identity.provider,
                submission_key=execution.identity.key,
                request_fingerprint=execution.identity.fingerprint,
                pipeline_id=execution.target.pipeline_id,
                models=list(execution.target.models),
                pipeline_digest=execution.target.pipeline_digest,
                pipeline=execution.target.pipeline,
                status=execution.status,
                invocation=execution.invocation,
                protected_context=sealed_context,
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
            .returning(ExecutionRow.id)
        )
        return inserted is not None

    async def find(self, identity: SubmissionIdentity) -> ExecutionRecord | None:
        """Read the row with a provider and key.

        Args:
            identity: Replay identity of the submission.

        Returns:
            The record, or None.
        """
        row = await self._session.scalar(
            select(ExecutionRow).where(
                ExecutionRow.provider == identity.provider,
                ExecutionRow.submission_key == identity.key,
            )
        )
        return None if row is None else _record(row)

    async def get(self, execution_id: str, *, lock: bool = False) -> ExecutionRecord:
        """Read a row by ID, optionally locking it.

        Args:
            execution_id: ID of the execution.
            lock: Whether to lock the row until the transaction ends.

        Returns:
            The record.

        Raises:
            UnknownExecutionError: The ID is not a UUID or no row has it.
        """
        return _record(await self._row(execution_id, lock=lock))

    async def get_many(self, execution_ids: Collection[str]) -> list[ExecutionRecord]:
        """Read the rows that exist, in the order requested.

        Args:
            execution_ids: IDs to read; ones that are not UUIDs are skipped.

        Returns:
            The records found.
        """
        valid = [value for value in execution_ids if _is_uuid(value)]
        if not valid:
            return []
        rows = await self._session.scalars(
            select(ExecutionRow).where(ExecutionRow.id.in_(valid))
        )
        found = {row.id: _record(row) for row in rows}
        return [found[value] for value in valid if value in found]

    async def with_work(self) -> list[ExecutionRecord]:
        """Read every row with pending work or a lease."""
        rows = await self._session.scalars(
            select(ExecutionRow).where(
                or_(
                    ExecutionRow.work_kind.is_not(None),
                    ExecutionRow.fencing_token.is_not(None),
                )
            )
        )
        return [_record(row) for row in rows]

    async def sealed_context(self, execution_id: str) -> bytes:
        """Read a row's encrypted provider context.

        Args:
            execution_id: ID of the execution.

        Returns:
            The stored bytes.
        """
        return bytes((await self._row(execution_id)).protected_context)

    async def save(
        self, execution: ExecutionRecord, *, sealed_context: bytes | None = None
    ) -> None:
        """Copy a record's changing fields onto its row and flush.

        Args:
            execution: The new state.
            sealed_context: New encrypted context; None keeps the stored one.
        """
        row = await self._row(execution.id)
        row.status = execution.status
        row.work_kind = execution.work_kind
        row.next_attempt_at = execution.next_attempt_at
        row.processing_attempts = execution.processing_attempts
        row.processing_rejections = execution.processing_rejections
        row.finish_attempts = execution.finish_attempts
        row.finish_mutation_started = execution.finish_mutation_started
        row.lease_owner = execution.lease_owner
        row.fencing_token = execution.fencing_token
        row.lease_expires_at = execution.lease_expires_at
        row.result = execution.result
        row.error_code = execution.error.code if execution.error is not None else None
        row.error_message = (
            execution.error.message if execution.error is not None else None
        )
        row.updated_at = execution.updated_at
        if sealed_context is not None:
            row.protected_context = sealed_context
        await self._session.flush()

    async def _row(self, execution_id: str, *, lock: bool = False) -> ExecutionRow:
        """Read a row by ID, optionally locking it.

        Args:
            execution_id: ID of the execution.
            lock: Whether to lock the row until the transaction ends.

        Returns:
            The row.

        Raises:
            UnknownExecutionError: The ID is not a UUID or no row has it.
        """
        if not _is_uuid(execution_id):
            raise UnknownExecutionError(execution_id)
        statement = select(ExecutionRow).where(ExecutionRow.id == execution_id)
        if lock:
            # Refresh what an earlier read of this transaction left in the session,
            # or a write committed while waiting for the lock would be overwritten.
            statement = statement.with_for_update().execution_options(
                populate_existing=True
            )
        row = await self._session.scalar(statement)
        if row is None:
            raise UnknownExecutionError(execution_id)
        return row


def _record(row: ExecutionRow) -> ExecutionRecord:
    """Convert a row to a domain record, without its provider context.

    Args:
        row: Row to convert.

    Returns:
        The record.

    Raises:
        ValueError: The row has an error code without a message.
    """
    error = None
    if row.error_code is not None:
        if row.error_message is None:
            raise ValueError(f"execution {row.id} has an error code without a message")
        error = ExecutionError(row.error_code, row.error_message)
    return ExecutionRecord(
        id=row.id,
        identity=SubmissionIdentity(
            provider=row.provider,
            key=row.submission_key,
            fingerprint=row.request_fingerprint,
        ),
        target=ExecutionTarget(
            pipeline_id=row.pipeline_id,
            models=tuple(row.models),
            pipeline_digest=row.pipeline_digest,
            pipeline=row.pipeline,
        ),
        status=row.status,
        invocation=row.invocation,
        protected_context={},
        deadline_at=row.deadline_at,
        terminal_report_deadline_at=row.terminal_report_deadline_at,
        work_kind=row.work_kind,
        next_attempt_at=row.next_attempt_at,
        processing_attempts=row.processing_attempts,
        processing_rejections=row.processing_rejections,
        finish_attempts=row.finish_attempts,
        finish_mutation_started=row.finish_mutation_started,
        lease_owner=row.lease_owner,
        fencing_token=row.fencing_token,
        lease_expires_at=row.lease_expires_at,
        result=row.result,
        error=error,
        created_at=row.created_at,
        updated_at=row.updated_at,
        priority=row.priority,
    )


def _is_uuid(value: str) -> bool:
    """Return whether a value is a UUID in canonical form.

    Args:
        value: Candidate execution ID.

    Returns:
        True if the value round-trips through UUID unchanged.
    """
    try:
        return str(UUID(value)) == value
    except ValueError:
        return False
