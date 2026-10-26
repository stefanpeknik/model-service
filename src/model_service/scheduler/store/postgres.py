"""Persist executions in PostgreSQL through operation-scoped SQLAlchemy transactions.

Row locks protect record changes; per-model advisory locks serialize capacity checks.
Each operation runs in its own transaction on a pooled connection, and database time
governs leases.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import datetime, timedelta
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from psycopg import AsyncConnection
from psycopg import Error as PsycopgError
from sqlalchemy import and_, delete, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from model_service.scheduler.contracts import (
    ExecutionError,
    ExecutionStatus,
    Operation,
    ProviderName,
    ResultUnavailableError,
    ScheduleConflictError,
    UnknownExecutionError,
)
from model_service.scheduler.domain import (
    Admission,
    AdmissionResult,
    ExecutionRecord,
    InferenceSucceeded,
    RenewDirective,
    RetryPolicy,
    Settlement,
    SubmissionIdentity,
    WorkClaim,
    WorkKind,
)
from model_service.scheduler.store.models import (
    DEFAULT_SCHEMA,
    ExecutionRow,
    InferencePayloadRow,
)
from model_service.scheduler.store.repository import ExecutionRepositoryUnavailableError
from model_service.scheduler.store.schema import initialize_execution_schema
from model_service.scheduler.transitions import ExecutionTransitions


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Collection

    from model_service.scheduler.store.protected_context import ProtectedContextCipher


class PostgresExecutionRepository:
    """Provide durable replay-safe admission, leased claims, and recovery using the shared state machine."""

    def __init__(
        self,
        dsn: str,
        *,
        cipher: ProtectedContextCipher,
        schema: str = DEFAULT_SCHEMA,
        per_model_limit: int = 1,
        per_model_inference_limit: int = 32,
        inference_retention: timedelta = timedelta(hours=24),
        processing_policy: RetryPolicy | None = None,
        finish_policy: RetryPolicy | None = None,
        pool_size: int = 10,
    ) -> None:
        """Validate schema and capacity, configure a pooled async engine, and retain context encryption and retry policies. No database connection opens here."""
        if not schema or not schema.replace("_", "a").isalnum():
            raise ValueError("PostgreSQL schema must be an identifier")
        if per_model_limit <= 0 or per_model_inference_limit <= 0:
            raise ValueError("per-model limits must be positive")
        self._cipher = cipher
        self._schema = schema
        self._dsn = dsn

        async def connect() -> AsyncConnection:
            # Accept existing libpq DSNs as well as PostgreSQL URLs.
            """Open a psycopg connection from either a libpq DSN or PostgreSQL URL for one engine checkout."""
            return await AsyncConnection.connect(dsn)

        # A new connection per transaction (fork plus SCRAM authentication) made
        # PostgreSQL the bottleneck when many tiles are scheduled.
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
        self._limits = {
            Operation.BUILDER: per_model_limit,
            Operation.INFERENCE: per_model_inference_limit,
        }
        self._inference_retention = inference_retention
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
        await self._engine.dispose()

    async def initialize(self) -> None:
        """Apply pending Yoyo migrations under a schema-specific initialization lock."""
        await initialize_execution_schema(self._dsn, schema=self._schema)

    async def check_ready(self) -> None:
        """Query the execution table to confirm database connectivity and schema availability."""
        async with self._transaction() as session:
            await session.execute(select(ExecutionRow.id).limit(1))

    async def maintain(self) -> None:
        """Expire deadlines, recover abandoned claims and delete expired inference results."""
        async with self._transaction() as session:
            now = await self._database_now(session)
            await self._expire_deadlines(session, now)
            await self._recover_expired_leases(session, now)
            await self._delete_expired_inference(session, now)

    async def find_submission(
        self,
        identity: SubmissionIdentity,
    ) -> ExecutionRecord | None:
        """Read by provider and replay key, decrypt the record, and reject a different request fingerprint."""
        async with self._transaction() as session:
            row = await self._find_submission(session, identity)
            if row is None:
                return None
            execution = self._record(row)
            self._check_fingerprint(execution, identity)
            return execution

    async def admit(self, admission: Admission) -> AdmissionResult:
        """Encrypt protected context and atomically insert a queued record or return an identical concurrent replay."""
        encrypted_context = self._cipher.encrypt(admission.protected_context)
        async with self._transaction() as session:
            execution = self._transitions.admit(
                admission, str(uuid4()), await self._database_now(session)
            )
            row = await session.scalar(
                insert(ExecutionRow)
                .values(
                    id=execution.id,
                    operation=execution.operation,
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
                    created_at=execution.created_at,
                    updated_at=execution.updated_at,
                )
                .on_conflict_do_nothing(
                    index_elements=[ExecutionRow.provider, ExecutionRow.submission_key]
                )
                .returning(ExecutionRow)
            )
            if row is not None:
                if admission.payload is not None:
                    session.add(
                        InferencePayloadRow(
                            execution_id=execution.id, request=admission.payload
                        )
                    )
                    await session.flush()
                return AdmissionResult(self._record(row), created=True)
            existing = await self._find_submission(session, admission.identity)
            if existing is None:
                raise RuntimeError("conflicting admission disappeared")
            execution = self._record(existing)
            self._check_fingerprint(execution, admission.identity)
            return AdmissionResult(execution, created=False)

    async def get(self, execution_id: str) -> ExecutionRecord:
        """Read and decode a domain record, raising UnknownExecutionError for an absent ID."""
        async with self._transaction() as session:
            return self._record(await self._fetch(session, execution_id))

    async def get_many(self, execution_ids: Collection[str]) -> list[ExecutionRecord]:
        """Read existing records in the requested order, ignoring unknown IDs."""
        valid = [value for value in execution_ids if _is_uuid(value)]
        if not valid:
            return []
        async with self._transaction() as session:
            rows = await session.scalars(
                select(ExecutionRow).where(ExecutionRow.id.in_(valid))
            )
            found = {row.id: self._record(row) for row in rows}
        return [found[value] for value in valid if value in found]

    async def request_cancel(self, execution_id: str) -> ExecutionRecord:
        """Lock the execution row and persist cancellation with the old claim invalidated; terminal outcomes remain unchanged."""
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
        operations: Collection[Operation] | None = None,
    ) -> WorkClaim | None:
        """Run maintenance, then lease due provider work before capacity-limited processing using database time."""
        allowed = frozenset(Operation if operations is None else operations)
        async with self._transaction() as session:
            now = await self._database_now(session)
            await self._expire_deadlines(session, now)
            await self._recover_expired_leases(session, now)
            row = None
            if Operation.BUILDER in allowed:
                row = await self._select_finish(session, now)
            if row is None:
                rows = await self._select_processing(session, now, allowed, limit=1)
                row = rows[0] if rows else None
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
            if claimed.claim.operation is not Operation.INFERENCE:
                return claimed.claim
            payload = await session.scalar(
                select(InferencePayloadRow.request).where(
                    InferencePayloadRow.execution_id == row.id
                )
            )
            if payload is None:
                raise RuntimeError("inference execution has no stored request")
            return replace(claimed.claim, payload=payload)

    async def claim_inference(
        self, owner: str, lease_duration: timedelta, limit: int
    ) -> list[WorkClaim]:
        """Lease up to `limit` due inference attempts, with payloads, in one transaction."""
        if limit <= 0:
            return []
        async with self._transaction() as session:
            now = await self._database_now(session)
            await self._expire_deadlines(session, now)
            await self._recover_expired_leases(session, now)
            rows = await self._select_processing(
                session, now, frozenset({Operation.INFERENCE}), limit=limit
            )
            claims = []
            for row in rows:
                claimed = self._transitions.claim(
                    self._record(row),
                    owner=owner,
                    fencing_token=str(uuid4()),
                    lease_duration=lease_duration,
                    now=now,
                )
                self._apply_state(row, claimed.execution)
                claims.append(claimed.claim)
            if not claims:
                return []
            await session.flush()
            requests = await session.execute(
                select(
                    InferencePayloadRow.execution_id, InferencePayloadRow.request
                ).where(
                    InferencePayloadRow.execution_id.in_(
                        [claim.execution_id for claim in claims]
                    )
                )
            )
            payloads = dict(requests.tuples().all())
            return [
                replace(claim, payload=payloads[claim.execution_id]) for claim in claims
            ]

    async def inference_content(self, execution_id: str) -> bytes:
        """Return the stored response of a succeeded inference execution."""
        async with self._transaction() as session:
            row = await self._fetch(session, execution_id)
            content = None
            if (
                row.operation is Operation.INFERENCE
                and row.status is ExecutionStatus.SUCCEEDED
            ):
                content = await session.scalar(
                    select(InferencePayloadRow.response).where(
                        InferencePayloadRow.execution_id == execution_id
                    )
                )
            if content is None:
                raise ResultUnavailableError("execution has no inference response")
            return content

    async def renew(
        self,
        execution_id: str,
        fencing_token: str,
        lease_duration: timedelta,
    ) -> RenewDirective:
        """Lock a record, check its token and deadline, and commit a valid lease extension or stop decision."""
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
        """Lock and validate an attempt's outcome, committing deadline changes before raising a stale-claim rejection."""
        async with self._transaction() as session:
            row = await self._fetch(session, execution_id, for_update=True)
            decision = self._transitions.decide_settlement(
                self._record(row),
                fencing_token,
                settlement,
                await self._database_now(session),
            )
            if decision.execution is not None:
                await self._persist_state(
                    session,
                    row,
                    decision.execution,
                    response=(
                        settlement.content
                        if decision.rejection is None
                        and isinstance(settlement, InferenceSucceeded)
                        else None
                    ),
                )
        return decision.accepted_execution()

    async def mark_finish_mutation_started(
        self,
        execution_id: str,
        fencing_token: str,
    ) -> None:
        """Lock and validate the provider claim before committing its external-mutation marker."""
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
        """Lock the earliest eligible provider attempt with SKIP LOCKED, excluding expired work deadlines."""
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
        self,
        session: AsyncSession,
        now: datetime,
        operations: frozenset[Operation],
        *,
        limit: int,
    ) -> list[ExecutionRow]:
        """Up to `limit` oldest eligible processing rows within each model's capacity.

        Builders and raw inference have separate limits per model. A
        transaction-scoped advisory lock per (operation, model) serializes the
        capacity check across scheduler replicas.
        """
        eligible = (
            ExecutionRow.work_kind == WorkKind.PROCESSING,
            ExecutionRow.fencing_token.is_(None),
            ExecutionRow.next_attempt_at <= now,
            ExecutionRow.deadline_at > now,
            ExecutionRow.operation.in_(operations),
        )
        targets = await session.execute(
            select(ExecutionRow.operation, ExecutionRow.model_id)
            .where(*eligible)
            .group_by(ExecutionRow.operation, ExecutionRow.model_id)
            .order_by(func.min(ExecutionRow.created_at))
        )
        selected: list[ExecutionRow] = []
        for operation, model_id in targets.tuples():
            locked = await session.scalar(
                select(
                    func.pg_try_advisory_xact_lock(
                        func.hashtextextended(f"{operation.value}:{model_id}", 0)
                    )
                )
            )
            if not locked:
                continue
            active = await session.scalar(
                select(func.count())
                .select_from(ExecutionRow)
                .where(
                    ExecutionRow.model_id == model_id,
                    ExecutionRow.operation == operation,
                    ExecutionRow.work_kind == WorkKind.PROCESSING,
                    ExecutionRow.fencing_token.is_not(None),
                    ExecutionRow.lease_expires_at > now,
                )
            )
            assert active is not None
            wanted = min(self._limits[operation] - active, limit - len(selected))
            if wanted <= 0:
                continue
            candidates = await session.scalars(
                select(ExecutionRow)
                .where(
                    ExecutionRow.model_id == model_id,
                    ExecutionRow.operation == operation,
                    *eligible,
                )
                .order_by(ExecutionRow.next_attempt_at, ExecutionRow.created_at)
                .with_for_update(skip_locked=True)
                .limit(wanted)
            )
            selected.extend(candidates)
            if len(selected) >= limit:
                break
        return selected

    async def _recover_expired_leases(
        self, session: AsyncSession, now: datetime
    ) -> None:
        """Lock available expired claims with SKIP LOCKED and persist their recovery decisions."""
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
        """Lock overdue active executions and terminal reports, skipping busy rows, and apply deadline transitions."""
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

    async def _delete_expired_inference(
        self, session: AsyncSession, now: datetime
    ) -> None:
        """Delete finished inference executions (and their bytes) after retention."""
        expired = (
            select(ExecutionRow.id)
            .where(
                ExecutionRow.work_kind.is_(None),
                ExecutionRow.operation == Operation.INFERENCE,
                ExecutionRow.updated_at <= now - self._inference_retention,
            )
            .limit(1000)
            .with_for_update(skip_locked=True)
        )
        await session.execute(delete(ExecutionRow).where(ExecutionRow.id.in_(expired)))

    async def _persist_state(
        self,
        session: AsyncSession,
        row: ExecutionRow,
        execution: ExecutionRecord,
        *,
        response: bytes | None = None,
    ) -> ExecutionRecord:
        """Flush the new state within the caller's transaction; return a decoded snapshot.

        A finished inference keeps only its response bytes.
        """
        self._apply_state(row, execution)
        await session.flush()
        if execution.operation is Operation.INFERENCE and execution.status in {
            ExecutionStatus.SUCCEEDED,
            ExecutionStatus.FAILED,
            ExecutionStatus.CANCELLED,
        }:
            values: dict[str, bytes | None] = {"request": None}
            if response is not None:
                values["response"] = response
            await session.execute(
                update(InferencePayloadRow)
                .where(InferencePayloadRow.execution_id == row.id)
                .values(**values)
            )
        return self._record(row)

    @staticmethod
    def _apply_state(row: ExecutionRow, execution: ExecutionRecord) -> None:
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
        """Read a row by ID, optionally locking it, or raise UnknownExecutionError."""
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
        """Query the unique provider/key identity in the caller's session without comparing its fingerprint."""
        return await session.scalar(
            select(ExecutionRow).where(
                ExecutionRow.provider == identity.provider,
                ExecutionRow.submission_key == identity.key,
            )
        )

    @staticmethod
    async def _database_now(session: AsyncSession) -> datetime:
        """Read PostgreSQL clock_timestamp so workers share the database's lease clock."""
        result = await session.execute(select(func.clock_timestamp()))
        return result.scalar_one()

    @asynccontextmanager
    async def _transaction(self) -> AsyncIterator[AsyncSession]:
        """Own one session and transaction, committing on success and translating database errors into availability failures."""
        try:
            async with self._sessions.begin() as session:
                yield session
        except (SQLAlchemyError, PsycopgError) as error:
            raise ExecutionRepositoryUnavailableError(
                "execution repository is unavailable"
            ) from error

    def _record(self, row: ExecutionRow) -> ExecutionRecord:
        """Convert an ORM row to an immutable domain record and decrypt private provider context."""
        error = None
        if row.error_code is not None:
            assert row.error_message is not None
            error = ExecutionError(row.error_code, row.error_message)
        return ExecutionRecord(
            id=row.id,
            identity=SubmissionIdentity(
                provider=ProviderName(row.provider),
                key=row.submission_key,
                fingerprint=row.request_fingerprint,
            ),
            builder_id=row.builder_id,
            model_id=row.model_id,
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
            operation=row.operation,
        )

    @staticmethod
    def _check_fingerprint(
        execution: ExecutionRecord,
        identity: SubmissionIdentity,
    ) -> None:
        """Reject reuse of a provider/key identity with different request contents."""
        if execution.identity.fingerprint != identity.fingerprint:
            raise ScheduleConflictError(
                "submission identity was reused with a different request"
            )


def _is_uuid(value: str) -> bool:
    try:
        return str(UUID(value)) == value
    except ValueError:
        return False
