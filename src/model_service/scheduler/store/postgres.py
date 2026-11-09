"""Persist executions in PostgreSQL through operation-scoped SQLAlchemy transactions.

Row locks protect record changes; per-model advisory locks serialize capacity checks.
Each operation runs in its own transaction on a pooled connection, and database time
governs leases.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from psycopg import AsyncConnection
from psycopg import Error as PsycopgError
from sqlalchemy import and_, func, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from model_service.scheduler.contracts import (
    ExecutionError,
    ExecutionStatus,
    ScheduleConflictError,
    UnknownExecutionError,
)
from model_service.scheduler.domain import (
    Admission,
    AdmissionResult,
    ExecutionRecord,
    RenewDirective,
    RetryPolicy,
    Settlement,
    SubmissionIdentity,
    WorkClaim,
    WorkKind,
)
from model_service.scheduler.store.models import DEFAULT_SCHEMA, ExecutionRow
from model_service.scheduler.store.repository import (
    ExecutionRepository,
    ExecutionRepositoryUnavailableError,
)
from model_service.scheduler.store.schema import initialize_execution_schema
from model_service.scheduler.transitions import (
    ExecutionTransitions,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from model_service.scheduler.store.protected_context import ProtectedContextCipher


logger = logging.getLogger(__name__)


class PostgresExecutionRepository(ExecutionRepository):
    """Provide durable replay-safe admission, leased claims, and recovery using the shared state machine."""

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
            ValueError: schema is not an identifier or per_model_limit is not positive.
        """
        if not schema or not schema.replace("_", "a").isalnum():
            raise ValueError("PostgreSQL schema must be an identifier")
        if per_model_limit <= 0:
            raise ValueError("per_model_limit must be positive")
        self._cipher = cipher
        self._schema = schema
        self._dsn = dsn

        async def connect() -> AsyncConnection:
            # Accept existing libpq DSNs as well as PostgreSQL URLs.
            """Open a psycopg connection from either a libpq DSN or PostgreSQL URL for one engine checkout."""
            return await AsyncConnection.connect(dsn)

        # A new connection per transaction costs a fork plus SCRAM authentication.
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
        self._per_model_limit = per_model_limit
        self._transitions = ExecutionTransitions(
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
        async with self._transaction() as session:
            await session.execute(select(ExecutionRow.id).limit(1))

    async def maintain(self) -> None:
        """Expire deadlines and recover abandoned claims.

        Raises:
            ExecutionRepositoryUnavailableError: The database is unavailable.
        """
        async with self._transaction() as session:
            now = await self._database_now(session)
            await self._expire_deadlines(session, now)
            await self._recover_expired_leases(session, now)

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
        async with self._transaction() as session:
            row = await self._find_submission(session, identity)
            if row is None:
                return None
            execution = self._record(row)
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
            RuntimeError: The conflicting row disappeared before it could be read.
            ExecutionRepositoryUnavailableError: The database is unavailable.
        """
        encrypted_context = self._cipher.encrypt(admission.protected_context)
        execution_id = str(uuid4())
        async with self._transaction() as session:
            execution = self._transitions.admit(
                admission, execution_id, await self._database_now(session)
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
                return AdmissionResult(self._record(row), created=True)
            existing = await self._find_submission(session, admission.identity)
            if existing is None:
                raise RuntimeError("conflicting admission disappeared")
            execution = self._record(existing)
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
        async with self._transaction() as session:
            return self._record(await self._fetch(session, execution_id))

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
        async with self._transaction() as session:
            row = await self._fetch(session, execution_id, for_update=True)
            execution = self._record(row)
            cancelled = self._transitions.cancel(
                execution, await self._database_now(session)
            )
            if cancelled is execution:
                return execution
            return await self._persist_state(session, row, cancelled)

    async def claim_due(
        self,
        owner: str,
        lease_duration: timedelta,
    ) -> WorkClaim | None:
        """Run maintenance, then lease due provider work before capacity-limited processing using database time.

        Args:
            owner: ID of the claiming worker.
            lease_duration: How long the claim stays valid without renewal.

        Returns:
            The claim, or None when nothing is due or the selected row's context is
            unreadable.

        Raises:
            ExecutionRepositoryUnavailableError: The database is unavailable.
        """
        async with self._transaction() as session:
            now = await self._database_now(session)
            await self._expire_deadlines(session, now)
            await self._recover_expired_leases(session, now)
            row = await self._select_finish(session, now)
            if row is None:
                row = await self._select_processing(session, now)
            if row is None:
                return None
            execution = await self._claimable(session, row, now)
            if execution is None:
                return None
            claimed = self._transitions.claim(
                execution,
                owner=owner,
                fencing_token=str(uuid4()),
                lease_duration=lease_duration,
                now=now,
            )
            await self._persist_state(session, row, claimed.execution)
            return claimed.claim

    async def renew(
        self,
        execution_id: str,
        fencing_token: str,
        lease_duration: timedelta,
    ) -> RenewDirective:
        """Lock a record, check its token and deadline, and commit a valid lease extension or stop decision.

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
        async with self._transaction() as session:
            row = await self._fetch(session, execution_id, for_update=True)
            decision = self._transitions.decide_renewal(
                self._record(row),
                fencing_token=fencing_token,
                lease_duration=lease_duration,
                now=await self._database_now(session),
            )
            if decision.execution is not None:
                await self._persist_state(session, row, decision.execution)
            return decision.directive

    async def settle(
        self,
        execution_id: str,
        fencing_token: str,
        settlement: Settlement,
    ) -> ExecutionRecord:
        """Lock and validate an attempt's outcome, committing deadline changes before raising a stale-claim rejection.

        Args:
            execution_id: ID of the claimed execution.
            fencing_token: Token of the settling attempt.
            settlement: Outcome of the attempt.

        Returns:
            The record after the outcome.

        Raises:
            UnknownExecutionError: No execution has this ID.
            StaleWorkClaimError: The token or lease is no longer current, the deadline passed, or the outcome does not fit the work.
            ExecutionRepositoryUnavailableError: The database is unavailable.
        """
        async with self._transaction() as session:
            row = await self._fetch(session, execution_id, for_update=True)
            decision = self._transitions.decide_settlement(
                self._record(row),
                fencing_token,
                settlement,
                await self._database_now(session),
            )
            if decision.execution is not None:
                await self._persist_state(session, row, decision.execution)
        return decision.accepted_execution()

    async def mark_finish_mutation_started(
        self,
        execution_id: str,
        fencing_token: str,
    ) -> None:
        """Lock and validate the provider claim before committing its external-mutation marker.

        Args:
            execution_id: ID of the claimed execution.
            fencing_token: Token of the provider attempt.

        Raises:
            UnknownExecutionError: No execution has this ID.
            StaleWorkClaimError: The claim is not provider work or is no longer current.
            ExecutionRepositoryUnavailableError: The database is unavailable.
        """
        async with self._transaction() as session:
            row = await self._fetch(session, execution_id, for_update=True)
            marked = self._transitions.mark_finish_mutation_started(
                self._record(row),
                fencing_token,
                await self._database_now(session),
            )
            await self._persist_state(session, row, marked)

    async def _select_finish(
        self, session: AsyncSession, now: datetime
    ) -> ExecutionRow | None:
        """Lock the earliest eligible provider attempt with SKIP LOCKED, excluding expired work deadlines.

        Args:
            session: Session of the caller's transaction.
            now: Current database time.

        Returns:
            The highest-priority due provider row, or None.
        """
        reports = (
            WorkKind.PROVIDER_FAILURE_REPORT,
            WorkKind.PROVIDER_CANCELLATION_REPORT,
        )
        return await session.scalar(
            select(ExecutionRow)
            .where(
                ExecutionRow.work_kind.in_((WorkKind.PROVIDER_FINISH, *reports)),
                ExecutionRow.fencing_token.is_(None),
                ExecutionRow.next_attempt_at <= now,
                or_(
                    and_(
                        ExecutionRow.work_kind == WorkKind.PROVIDER_FINISH,
                        ExecutionRow.deadline_at > now,
                    ),
                    and_(
                        ExecutionRow.work_kind.in_(reports),
                        func.coalesce(
                            ExecutionRow.terminal_report_deadline_at,
                            ExecutionRow.deadline_at,
                        )
                        > now,
                    ),
                ),
            )
            .order_by(
                ExecutionRow.priority.desc(),
                ExecutionRow.next_attempt_at,
                ExecutionRow.created_at,
            )
            .with_for_update(skip_locked=True)
            .limit(1)
        )

    async def _select_processing(
        self, session: AsyncSession, now: datetime
    ) -> ExecutionRow | None:
        """Lock the oldest eligible processing row whose model is below its capacity.

        A transaction-scoped advisory lock per model serializes the capacity check
        across scheduler replicas; a model whose lock is held elsewhere is skipped.

        Args:
            session: Session of the caller's transaction.
            now: Current database time.

        Returns:
            The locked row, by model priority and age, or None.
        """
        eligible = (
            ExecutionRow.work_kind == WorkKind.PROCESSING,
            ExecutionRow.fencing_token.is_(None),
            ExecutionRow.next_attempt_at <= now,
            ExecutionRow.deadline_at > now,
        )
        models = await session.scalars(
            select(ExecutionRow.model_id)
            .where(*eligible)
            .group_by(ExecutionRow.model_id)
            .order_by(
                func.max(ExecutionRow.priority).desc(),
                func.min(ExecutionRow.created_at),
            )
        )
        for model_id in models:
            locked = await session.scalar(
                select(
                    func.pg_try_advisory_xact_lock(func.hashtextextended(model_id, 0))
                )
            )
            if not locked:
                continue
            active = (
                await session.execute(
                    select(func.count())
                    .select_from(ExecutionRow)
                    .where(
                        ExecutionRow.model_id == model_id,
                        ExecutionRow.work_kind == WorkKind.PROCESSING,
                        ExecutionRow.fencing_token.is_not(None),
                        ExecutionRow.lease_expires_at > now,
                    )
                )
            ).scalar_one()
            if active >= self._per_model_limit:
                continue
            candidate = await session.scalar(
                select(ExecutionRow)
                .where(ExecutionRow.model_id == model_id, *eligible)
                .order_by(
                    ExecutionRow.priority.desc(),
                    ExecutionRow.next_attempt_at,
                    ExecutionRow.created_at,
                )
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            if candidate is not None:
                return candidate
        return None

    async def _recover_expired_leases(
        self, session: AsyncSession, now: datetime
    ) -> None:
        """Lock available expired claims with SKIP LOCKED and persist their recovery decisions.

        Args:
            session: Session of the caller's transaction.
            now: Current database time.
        """
        rows = await session.scalars(
            select(ExecutionRow)
            .where(
                ExecutionRow.fencing_token.is_not(None),
                ExecutionRow.lease_expires_at <= now,
            )
            .with_for_update(skip_locked=True)
        )
        for row in rows:
            recovered = self._transitions.recover_expired(self._record(row), now)
            await self._persist_state(session, row, recovered)

    async def _expire_deadlines(self, session: AsyncSession, now: datetime) -> None:
        """Lock overdue active executions and terminal reports, skipping busy rows, and apply deadline transitions.

        Args:
            session: Session of the caller's transaction.
            now: Current database time.
        """
        rows = await session.scalars(
            select(ExecutionRow)
            .where(
                # Implied by the conditions below; lets the planner use the
                # partial deadline indexes instead of scanning finished rows.
                ExecutionRow.work_kind.is_not(None),
                or_(
                    ExecutionRow.deadline_at <= now,
                    ExecutionRow.terminal_report_deadline_at <= now,
                ),
                or_(
                    and_(
                        ExecutionRow.status.not_in(
                            (
                                ExecutionStatus.SUCCEEDED,
                                ExecutionStatus.FAILED,
                                ExecutionStatus.CANCELLED,
                            )
                        ),
                        ExecutionRow.deadline_at <= now,
                    ),
                    and_(
                        ExecutionRow.status.in_(
                            (ExecutionStatus.FAILED, ExecutionStatus.CANCELLED)
                        ),
                        ExecutionRow.work_kind.in_(
                            (
                                WorkKind.PROVIDER_FAILURE_REPORT,
                                WorkKind.PROVIDER_CANCELLATION_REPORT,
                            )
                        ),
                        func.coalesce(
                            ExecutionRow.terminal_report_deadline_at,
                            ExecutionRow.deadline_at,
                        )
                        <= now,
                    ),
                ),
            )
            .with_for_update(skip_locked=True)
        )
        for row in rows:
            await self._persist_state(
                session, row, self._transitions.deadline(self._record(row), now)
            )

    async def _persist_state(
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
        self._apply_state(row, execution)
        if execution.work_kind is None:
            # Nothing will use the job's credential again.
            row.protected_context = self._cipher.encrypt({})
        await session.flush()
        return self._record(row)

    @staticmethod
    def _apply_state(row: ExecutionRow, execution: ExecutionRecord) -> None:
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

    async def _fetch(
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
    async def _database_now(session: AsyncSession) -> datetime:
        """Read PostgreSQL clock_timestamp so workers share the database's lease clock.

        Args:
            session: Session of the caller's transaction.

        Returns:
            The database's current time.
        """
        result = await session.execute(select(func.clock_timestamp()))
        return result.scalar_one()

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[AsyncSession]:
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

    async def _claimable(
        self, session: AsyncSession, row: ExecutionRow, now: datetime
    ) -> ExecutionRecord | None:
        """Decode a row with its provider context for claiming.

        A record that cannot be read fails, not the queue: it is ended in place, and
        its stored bytes stay in case the key was wrong.

        Args:
            session: Session of the caller's transaction.
            row: Locked candidate row.
            now: Current database time.

        Returns:
            The record with its decrypted context, or None if the context is unreadable.
        """
        try:
            return self._record(row, protected=True)
        except ValueError:  # Includes ProtectedContextError: a bad envelope or key.
            logger.error("execution %s has an unreadable provider context", row.id)
            # Not _persist_state: the stored bytes stay, in case the key was wrong.
            self._apply_state(row, self._transitions.unreadable(self._record(row), now))
            await session.flush()
            return None

    def _record(self, row: ExecutionRow, *, protected: bool = False) -> ExecutionRecord:
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
                self._cipher.decrypt(row.protected_context) if protected else {}
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
