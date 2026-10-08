"""Persist executions in PostgreSQL through operation-scoped SQLAlchemy transactions.

Row locks protect record changes; workflow advisory locks serialize capacity checks.
Each operation opens its own connection using NullPool, and database time governs
leases.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

from psycopg import AsyncConnection
from psycopg import Error as PsycopgError
from sqlalchemy import and_, func, or_, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from execution_scheduler.contracts import (
    ExecutionError,
    ExecutionStatus,
    ScheduleConflictError,
    UnknownExecutionError,
)
from execution_scheduler.domain import (
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
from execution_scheduler.persistence.models import DEFAULT_SCHEMA, ExecutionRow
from execution_scheduler.persistence.schema import initialize_execution_schema
from execution_scheduler.repository import ExecutionRepositoryUnavailableError
from execution_scheduler.transitions import ExecutionTransitions


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from execution_scheduler.persistence.protected_context import ProtectedContextCipher


class PostgresExecutionRepository:
    """Provide durable replay-safe admission, leased claims, and recovery using the shared state machine."""

    def __init__(
        self,
        dsn: str,
        *,
        cipher: ProtectedContextCipher,
        schema: str = DEFAULT_SCHEMA,
        per_workflow_limit: int = 1,
        processing_policy: RetryPolicy | None = None,
        finish_policy: RetryPolicy | None = None,
    ) -> None:
        """Validate schema and capacity, configure an unpooled async engine, and retain context encryption and retry policies. No database connection opens here.

        Args:
            dsn: PostgreSQL connection settings as a URL or libpq connection string.
            cipher: Configured authenticated cipher for protected provider context at
                rest.
            schema: Validated PostgreSQL schema name, or a separate isolated schema for
                the test.
            per_workflow_limit: Maximum simultaneous leased processing attempts per
                workflow.
            processing_policy: Attempt/backoff limits for workflow computation.
            finish_policy: Attempt/backoff limits for publication and terminal
                reporting.

        Raises:
            ValueError: schema is empty or contains unsupported identifier characters,
                or per_workflow_limit is nonpositive.
        """
        if not schema or not schema.replace("_", "a").isalnum():
            raise ValueError("PostgreSQL schema must be an identifier")
        if per_workflow_limit <= 0:
            raise ValueError("per_workflow_limit must be positive")
        self._cipher = cipher
        self._schema = schema

        async def connect() -> AsyncConnection:
            # Accept existing libpq DSNs as well as PostgreSQL URLs.
            """Open a psycopg connection from either a libpq DSN or PostgreSQL URL for one engine checkout.

            Returns:
                AsyncConnection: New psycopg connection for this database operation.
            """
            return await AsyncConnection.connect(dsn)

        engine = create_async_engine(
            "postgresql+psycopg://",
            async_creator=connect,
            poolclass=NullPool,
            hide_parameters=True,
            execution_options={"schema_translate_map": {DEFAULT_SCHEMA: schema}},
        )
        self._sessions = async_sessionmaker(engine, expire_on_commit=False)
        self._per_workflow_limit = per_workflow_limit
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

    async def initialize(self) -> None:
        """Create or add compatible schema elements inside a transaction guarded by a schema advisory lock.

        Returns:
            None: No return value.

        Raises:
            ExecutionRepositoryUnavailableError: Storage cannot create or upgrade the
                execution schema in its initialization transaction.
        """
        async with self._transaction() as session:
            await initialize_execution_schema(session, schema=self._schema)

    async def check_ready(self) -> None:
        """Query the execution table to confirm database connectivity and schema availability.

        Returns:
            None: No return value.

        Raises:
            ExecutionRepositoryUnavailableError: Database access or the storage
                transaction fails.
        """
        async with self._transaction() as session:
            await session.execute(select(ExecutionRow.id).limit(1))

    async def maintain(self) -> None:
        """Expire deadlines and recover abandoned claims in one database-time transaction.

        Returns:
            None: No return value.

        Raises:
            ExecutionRepositoryUnavailableError: Database access or the storage
                transaction fails.
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
        async with self._transaction() as session:
            row = await self._find_submission(session, identity)
            if row is None:
                return None
            execution = self._record(row)
            self._check_fingerprint(execution, identity)
            return execution

    async def admit(self, admission: Admission) -> AdmissionResult:
        """Encrypt protected context and atomically insert a queued record or return an identical concurrent replay.

        The unique provider/key constraint resolves races. A conflicting fingerprint
        raises ScheduleConflictError; transaction exit commits before this method
        returns.

        Args:
            admission: Prepared submission containing replay identity, invocation,
                deadlines, and private context.

        Returns:
            AdmissionResult: The admitted execution and whether this operation created
                it.

        Raises:
            ExecutionRepositoryUnavailableError: The database transaction fails.
            ScheduleConflictError: The same provider submission key has a different
                request fingerprint.
            RuntimeError: A conflicting submission disappears before it can be loaded.
        """
        encrypted_context = self._cipher.encrypt(admission.protected_context)
        async with self._transaction() as session:
            execution = self._transitions.admit(
                admission, str(uuid4()), await self._database_now(session)
            )
            row = await session.scalar(
                insert(ExecutionRow)
                .values(
                    id=execution.id,
                    provider=execution.identity.provider,
                    submission_key=execution.identity.key,
                    request_fingerprint=execution.identity.fingerprint,
                    workflow_id=execution.workflow_id,
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
            execution_id: Durable execution identifier, not an external provider job ID.

        Returns:
            ExecutionRecord: Internal durable execution snapshot, including work and
                ownership fields.

        Raises:
            ExecutionRepositoryUnavailableError: Database access or the storage
                transaction fails.
            UnknownExecutionError: The execution ID is absent from storage.
        """
        async with self._transaction() as session:
            return self._record(await self._fetch(session, execution_id))

    async def request_cancel(self, execution_id: str) -> ExecutionRecord:
        """Lock the execution row and persist cancellation with the old claim invalidated; terminal outcomes remain unchanged.

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
        async with self._transaction() as session:
            now = await self._database_now(session)
            await self._expire_deadlines(session, now)
            await self._recover_expired_leases(session, now)
            row = await self._select_finish(session, now)
            if row is None:
                row = await self._select_processing(session, now)
            if row is None:
                return None
            claimed = self._transitions.claim(
                self._record(row),
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
            session: Operation-scoped SQLAlchemy session inside the caller's
                transaction.
            now: Aware current time supplied by the repository or validation clock.

        Returns:
            ExecutionRow | None: Locked earliest eligible provider row, or None when no
                unlocked provider work is due.
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
            .order_by(ExecutionRow.next_attempt_at, ExecutionRow.created_at)
            .with_for_update(skip_locked=True)
            .limit(1)
        )

    async def _select_processing(
        self, session: AsyncSession, now: datetime
    ) -> ExecutionRow | None:
        """Choose eligible processing within workflow capacity using advisory and row locks.

        Workflows are considered by oldest eligible creation time. Locked or full
        workflows are skipped; concurrent claims do not guarantee global FIFO order.

        Args:
            session: Operation-scoped SQLAlchemy session inside the caller's
                transaction.
            now: Aware current time supplied by the repository or validation clock.

        Returns:
            ExecutionRow | None: Locked eligible processing row, or None when candidates
                are absent, locked, or at capacity.
        """
        eligible = (
            ExecutionRow.work_kind == WorkKind.PROCESSING,
            ExecutionRow.fencing_token.is_(None),
            ExecutionRow.next_attempt_at <= now,
            ExecutionRow.deadline_at > now,
        )
        workflows = await session.scalars(
            select(ExecutionRow.workflow_id)
            .where(*eligible)
            .group_by(ExecutionRow.workflow_id)
            .order_by(func.min(ExecutionRow.created_at))
        )
        for workflow_id in workflows:
            locked = await session.scalar(
                select(
                    func.pg_try_advisory_xact_lock(
                        func.hashtextextended(workflow_id, 0)
                    )
                )
            )
            if not locked:
                continue
            active = await session.scalar(
                select(func.count())
                .select_from(ExecutionRow)
                .where(
                    ExecutionRow.workflow_id == workflow_id,
                    ExecutionRow.work_kind == WorkKind.PROCESSING,
                    ExecutionRow.fencing_token.is_not(None),
                    ExecutionRow.lease_expires_at > now,
                )
            )
            assert active is not None
            if active >= self._per_workflow_limit:
                continue
            candidate = await session.scalar(
                select(ExecutionRow)
                .where(ExecutionRow.workflow_id == workflow_id, *eligible)
                .order_by(ExecutionRow.next_attempt_at, ExecutionRow.created_at)
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
            session: Operation-scoped SQLAlchemy session inside the caller's
                transaction.
            now: Aware current time supplied by the repository or validation clock.

        Returns:
            None: No return value.
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
            session: Operation-scoped SQLAlchemy session inside the caller's
                transaction.
            now: Aware current time supplied by the repository or validation clock.

        Returns:
            None: No return value.
        """
        rows = await session.scalars(
            select(ExecutionRow)
            .where(
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
                )
            )
            .with_for_update(skip_locked=True)
        )
        for row in rows:
            await self._persist_state(
                session, row, self._transitions.deadline(self._record(row), now)
            )

    async def _persist_state(
        self, session: AsyncSession, row: ExecutionRow, execution: ExecutionRecord
    ) -> ExecutionRecord:
        """Copy mutable state into an ORM row and flush within the caller's transaction; return a decoded snapshot.

        Args:
            session: Operation-scoped SQLAlchemy session inside the caller's
                transaction.
            row: Database row whose state is being decoded, checked, or updated.
            execution: Current immutable execution record; transitions return a
                replacement without mutating it.

        Returns:
            ExecutionRecord: Internal durable execution snapshot, including work and
                ownership fields.
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
        await session.flush()
        return self._record(row)

    async def _fetch(
        self, session: AsyncSession, execution_id: str, *, for_update: bool = False
    ) -> ExecutionRow:
        """Read a row by ID, optionally locking it, or raise UnknownExecutionError.

        Args:
            session: Operation-scoped SQLAlchemy session inside the caller's
                transaction.
            execution_id: Durable execution identifier, not an external provider job ID.
            for_update: Whether to lock the row for a state mutation in the current
                transaction.

        Returns:
            ExecutionRow: ORM row retained within the current persistence operation.

        Raises:
            UnknownExecutionError: The execution ID is absent from storage.
        """
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
            session: Operation-scoped SQLAlchemy session inside the caller's
                transaction.
            identity: Provider-scoped replay key and request fingerprint.

        Returns:
            ExecutionRow | None: Matching provider/key row, or None if no submission
                exists.
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
            session: Operation-scoped SQLAlchemy session inside the caller's
                transaction.

        Returns:
            datetime: Timezone-aware PostgreSQL wall clock used to evaluate leases and
                deadlines.
        """
        result = await session.execute(select(func.clock_timestamp()))
        return result.scalar_one()

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[AsyncSession]:
        """Own one session and transaction, committing on success and translating database errors into availability failures.

        Yields:
            AsyncSession: Operation-scoped session committed or rolled back on context
                exit.

        Raises:
            ExecutionRepositoryUnavailableError: Database access or the storage
                transaction fails.
        """
        try:
            async with self._sessions.begin() as session:
                yield session
        except (SQLAlchemyError, PsycopgError) as error:
            raise ExecutionRepositoryUnavailableError(
                "execution repository is unavailable"
            ) from error

    def _record(self, row: ExecutionRow) -> ExecutionRecord:
        """Convert an ORM row to an immutable domain record and decrypt private provider context.

        Args:
            row: Database row whose state is being decoded, checked, or updated.

        Returns:
            ExecutionRecord: Internal durable execution snapshot, including work and
                ownership fields.
        """
        error = None
        if row.error_code is not None:
            assert row.error_message is not None
            error = ExecutionError(row.error_code, row.error_message)
        return ExecutionRecord(
            id=row.id,
            identity=SubmissionIdentity(
                provider=row.provider,
                key=row.submission_key,
                fingerprint=row.request_fingerprint,
            ),
            workflow_id=row.workflow_id,
            status=row.status,
            invocation=row.invocation,
            protected_context=self._cipher.decrypt(row.protected_context),
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
        )

    @staticmethod
    def _check_fingerprint(
        execution: ExecutionRecord,
        identity: SubmissionIdentity,
    ) -> None:
        """Reject reuse of a provider/key identity with different request contents.

        Args:
            execution: Current immutable execution record; transitions return a
                replacement without mutating it.
            identity: Provider-scoped replay key and request fingerprint.

        Returns:
            None: No return value.

        Raises:
            ScheduleConflictError: A provider submission key already exists with
                different request contents.
        """
        if execution.identity.fingerprint != identity.fingerprint:
            raise ScheduleConflictError(
                "submission identity was reused with a different request"
            )
