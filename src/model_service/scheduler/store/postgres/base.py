"""The PostgreSQL repository's connection, transactions and row mapping.

Row locks protect record changes; per-model advisory locks serialize capacity checks.
Each operation runs in its own transaction on a pooled connection, and database time
governs leases.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from uuid import UUID

from psycopg import AsyncConnection
from psycopg import Error as PsycopgError
from sqlalchemy import func, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from model_service.scheduler.internal_state import (
    ExecutionRecord,
    RetryPolicy,
)
from model_service.scheduler.public import (
    ExecutionError,
    SubmissionIdentity,
    UnknownExecutionError,
)
from model_service.scheduler.store.migrate import initialize_execution_schema
from model_service.scheduler.store.repository import ExecutionRepositoryUnavailableError
from model_service.scheduler.store.rows import DEFAULT_SCHEMA, ExecutionRow
from model_service.scheduler.transitions import ExecutionTransitions


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from model_service.scheduler.store.protected_context import ProtectedContextCipher


class PostgresDatabase:
    """The connection pool, transactions and row mapping every group of operations uses.

    Attributes:
        cipher: Cipher for the stored provider context.
        per_model_limit: Concurrent builder executions allowed per model.
        transitions: The shared state machine.
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
        """Validate schema and capacity, configure a pooled async engine, and retain context encryption and retry policies.

        No database connection opens here.

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
        if not schema or not schema.replace("_", "a").isalnum():
            raise ValueError("PostgreSQL schema must be an identifier")
        if per_model_limit <= 0:
            raise ValueError("per-model limits must be positive")
        self.cipher = cipher
        self._schema = schema
        self._dsn = dsn

        async def connect() -> AsyncConnection:
            # Accept existing libpq DSNs as well as PostgreSQL URLs.
            """Open a psycopg connection from either a libpq DSN or PostgreSQL URL for one engine checkout."""
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
        self.per_model_limit = per_model_limit
        self.transitions = ExecutionTransitions(
            processing_policy
            or RetryPolicy(
                max_attempts=3,
                initial_backoff=timedelta(seconds=5),
                maximum_backoff=timedelta(minutes=1),
            ),
            finish_policy
            or RetryPolicy(
                max_attempts=3,
                initial_backoff=timedelta(seconds=5),
                maximum_backoff=timedelta(minutes=1),
            ),
        )

    async def aclose(self) -> None:
        """Dispose of the connection pool."""
        await self._engine.dispose()

    async def initialize(self) -> None:
        """Apply pending Yoyo migrations under a schema-specific initialization lock."""
        await initialize_execution_schema(self._dsn, schema=self._schema)

    async def check_ready(self) -> None:
        """Query the execution table to confirm database connectivity and schema availability.

        Raises:
            ExecutionRepositoryUnavailableError: The database is unavailable.
        """
        async with self.transaction() as session:
            await session.execute(select(ExecutionRow.id).limit(1))

    async def persist_state(
        self,
        session: AsyncSession,
        row: ExecutionRow,
        execution: ExecutionRecord,
    ) -> ExecutionRecord:
        """Flush the new state within the caller's transaction; return a decoded snapshot.

        Finished work drops its provider context, since nothing will use the job's
        credential again.

        Args:
            session: Session of the caller's transaction.
            row: Locked row to update.
            execution: New state.

        Returns:
            The record read back from the row, without its protected context.
        """
        self.apply_state(row, execution)
        if execution.work_kind is None:
            # Nothing will use the job's credential again.
            row.protected_context = self.cipher.encrypt({})
        await session.flush()
        return self.record(row)

    @staticmethod
    def apply_state(row: ExecutionRow, execution: ExecutionRecord) -> None:
        """Copy the mutable fields of a record onto its row without flushing.

        Args:
            row: Row to update.
            execution: New state.
        """
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

    async def fetch(
        self, session: AsyncSession, execution_id: str, *, for_update: bool = False
    ) -> ExecutionRow:
        """Read a row by ID, optionally locking it, or raise UnknownExecutionError.

        Args:
            session: Session of the caller's transaction.
            execution_id: ID of the execution.
            for_update: Whether to lock the row for the transaction.

        Returns:
            The row.

        Raises:
            UnknownExecutionError: The ID is not a UUID or no row has it.
        """
        if not _is_uuid(execution_id):
            raise UnknownExecutionError(execution_id)
        statement = select(ExecutionRow).where(ExecutionRow.id == execution_id)
        if for_update:
            statement = statement.with_for_update()
        row = await session.scalar(statement)
        if row is None:
            raise UnknownExecutionError(execution_id)
        return row

    @staticmethod
    async def database_now(session: AsyncSession) -> datetime:
        """Read PostgreSQL clock_timestamp so workers share the database's lease clock.

        Args:
            session: Session of the caller's transaction.

        Returns:
            The database's current time.
        """
        result = await session.execute(select(func.clock_timestamp()))
        return result.scalar_one()

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[AsyncSession]:
        """Own one session and transaction, committing on success and translating database errors into availability failures.

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

    def record(self, row: ExecutionRow, *, protected: bool = False) -> ExecutionRecord:
        """Convert an ORM row to an immutable domain record; only a claim decrypts the provider context.

        Args:
            row: Row to convert.
            protected: Whether to decrypt the provider context.

        Returns:
            The domain record, with an empty context unless protected is set.

        Raises:
            ValueError: protected is set and the context cannot be decrypted; includes ProtectedContextError.
            TypeError: protected is set and the context is not a JSON object.
        """
        error = None
        if row.error_code is not None:
            if row.error_message is None:
                raise ValueError(
                    f"execution {row.id} has an error code without a message"
                )
            error = ExecutionError(row.error_code, row.error_message)
        return ExecutionRecord(
            id=row.id,
            identity=SubmissionIdentity(
                provider=row.provider,
                key=row.submission_key,
                fingerprint=row.request_fingerprint,
            ),
            builder_id=row.builder_id,
            model_id=row.model_id,
            status=row.status,
            invocation=row.invocation,
            protected_context=(
                self.cipher.decrypt(row.protected_context) if protected else {}
            ),
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
