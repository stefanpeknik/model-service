from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, cast
from uuid import uuid4

from psycopg import AsyncConnection, sql
from psycopg import Error as PsycopgError
from psycopg.rows import DictRow, dict_row
from psycopg.types.json import Jsonb

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
from execution_scheduler.persistence.schema import initialize_execution_schema
from execution_scheduler.repository import ExecutionRepositoryUnavailableError
from execution_scheduler.transitions import ExecutionTransitions
from workflow_execution.serialization import (
    invocation_from_dict,
    invocation_to_dict,
    result_from_dict,
    result_to_dict,
)


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from execution_scheduler.persistence.protected_context import ProtectedContextCipher


class PostgresExecutionRepository:
    def __init__(
        self,
        dsn: str,
        *,
        cipher: ProtectedContextCipher,
        schema: str = "model_service",
        per_workflow_limit: int = 1,
        processing_policy: RetryPolicy | None = None,
        finish_policy: RetryPolicy | None = None,
    ) -> None:
        if not schema or not schema.replace("_", "a").isalnum():
            raise ValueError("PostgreSQL schema must be an identifier")
        if per_workflow_limit <= 0:
            raise ValueError("per_workflow_limit must be positive")
        self._dsn = dsn
        self._cipher = cipher
        self._schema = schema
        self._table = sql.Identifier(schema, "executions")
        self._per_workflow_limit = per_workflow_limit
        self._processing_policy = processing_policy or RetryPolicy(
            max_attempts=3,
            initial_backoff=timedelta(seconds=5),
            maximum_backoff=timedelta(minutes=1),
        )
        self._finish_policy = finish_policy or RetryPolicy(
            max_attempts=3,
            initial_backoff=timedelta(seconds=5),
            maximum_backoff=timedelta(minutes=1),
        )
        self._transitions = ExecutionTransitions(
            self._processing_policy,
            self._finish_policy,
        )

    async def initialize(self) -> None:
        async with self._connection() as connection:
            await initialize_execution_schema(
                connection,
                schema=self._schema,
            )

    async def check_ready(self) -> None:
        async with self._connection() as connection:
            await connection.execute(
                sql.SQL("SELECT id FROM {} LIMIT 1").format(self._table)
            )

    async def maintain(self) -> None:
        async with self._connection() as connection:
            now = await self._database_now(connection)
            await self._expire_deadlines(connection, now)
            await self._recover_expired_leases(connection, now)

    async def find_submission(
        self,
        identity: SubmissionIdentity,
    ) -> ExecutionRecord | None:
        async with self._connection() as connection:
            cursor = await connection.execute(
                sql.SQL(
                    "SELECT * FROM {} WHERE provider = %s AND submission_key = %s"
                ).format(self._table),
                (identity.provider, identity.key),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        execution = self._record(row)
        self._check_fingerprint(execution, identity)
        return execution

    async def admit(self, admission: Admission) -> AdmissionResult:
        encrypted_context = self._cipher.encrypt(admission.protected_context)
        async with self._connection() as connection:
            now = await self._database_now(connection)
            execution = self._transitions.admit(
                admission,
                str(uuid4()),
                now,
            )
            cursor = await connection.execute(
                sql.SQL(
                    """
                    INSERT INTO {} (
                        id, provider, submission_key, request_fingerprint,
                        workflow_id, status, invocation, protected_context,
                        deadline_at, terminal_report_deadline_at, work_kind,
                        next_attempt_at, processing_attempts,
                        processing_rejections, finish_attempts,
                        finish_mutation_started, created_at,
                        updated_at
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                        %s, %s, %s, %s, %s, %s, %s
                    )
                    ON CONFLICT (provider, submission_key) DO NOTHING
                    RETURNING *
                    """
                ).format(self._table),
                (
                    execution.id,
                    execution.identity.provider,
                    execution.identity.key,
                    execution.identity.fingerprint,
                    execution.workflow_id,
                    execution.status,
                    Jsonb(invocation_to_dict(execution.invocation)),
                    encrypted_context,
                    execution.deadline_at,
                    execution.terminal_report_deadline_at,
                    execution.work_kind,
                    execution.next_attempt_at,
                    execution.processing_attempts,
                    execution.processing_rejections,
                    execution.finish_attempts,
                    execution.finish_mutation_started,
                    execution.created_at,
                    execution.updated_at,
                ),
            )
            row = await cursor.fetchone()
            if row is not None:
                return AdmissionResult(self._record(row), created=True)
            existing_cursor = await connection.execute(
                sql.SQL(
                    "SELECT * FROM {} WHERE provider = %s AND submission_key = %s"
                ).format(self._table),
                (admission.identity.provider, admission.identity.key),
            )
            existing = await existing_cursor.fetchone()
            if existing is None:
                raise RuntimeError("conflicting admission disappeared")
            execution = self._record(existing)
            self._check_fingerprint(execution, admission.identity)
            return AdmissionResult(execution, created=False)

    async def get(self, execution_id: str) -> ExecutionRecord:
        async with self._connection() as connection:
            row = await self._fetch(connection, execution_id)
        return self._record(row)

    async def request_cancel(self, execution_id: str) -> ExecutionRecord:
        async with self._connection() as connection:
            row = await self._fetch(connection, execution_id, for_update=True)
            execution = self._record(row)
            cancelled = self._transitions.cancel(
                execution,
                await self._database_now(connection),
            )
            if cancelled is execution:
                return execution
            return await self._persist_state(connection, cancelled)

    async def claim_due(
        self,
        owner: str,
        lease_duration: timedelta,
    ) -> WorkClaim | None:
        async with self._connection() as connection:
            now = await self._database_now(connection)
            await self._expire_deadlines(connection, now)
            await self._recover_expired_leases(connection, now)
            row = await self._select_finish(connection, now)
            if row is None:
                row = await self._select_processing(connection, now)
            if row is None:
                return None
            execution = self._record(row)
            token = str(uuid4())
            claimed = self._transitions.claim(
                execution,
                owner=owner,
                fencing_token=token,
                lease_duration=lease_duration,
                now=now,
            )
            await self._persist_state(connection, claimed.execution)
            return claimed.claim

    async def renew(
        self,
        execution_id: str,
        fencing_token: str,
        lease_duration: timedelta,
    ) -> RenewDirective:
        async with self._connection() as connection:
            row = await self._fetch(connection, execution_id, for_update=True)
            execution = self._record(row)
            now = await self._database_now(connection)
            decision = self._transitions.decide_renewal(
                execution,
                fencing_token=fencing_token,
                lease_duration=lease_duration,
                now=now,
            )
            if decision.execution is not None:
                await self._persist_state(connection, decision.execution)
            return decision.directive

    async def settle(
        self,
        execution_id: str,
        fencing_token: str,
        settlement: Settlement,
    ) -> ExecutionRecord:
        async with self._connection() as connection:
            row = await self._fetch(connection, execution_id, for_update=True)
            execution = self._record(row)
            now = await self._database_now(connection)
            decision = self._transitions.decide_settlement(
                execution,
                fencing_token,
                settlement,
                now,
            )
            if decision.execution is not None:
                await self._persist_state(connection, decision.execution)
        return decision.accepted_execution()

    async def mark_finish_mutation_started(
        self,
        execution_id: str,
        fencing_token: str,
    ) -> None:
        async with self._connection() as connection:
            row = await self._fetch(connection, execution_id, for_update=True)
            execution = self._record(row)
            marked = self._transitions.mark_finish_mutation_started(
                execution,
                fencing_token,
                await self._database_now(connection),
            )
            await self._persist_state(connection, marked)

    async def _select_finish(
        self,
        connection: AsyncConnection[DictRow],
        now: datetime,
    ) -> DictRow | None:
        cursor = await connection.execute(
            sql.SQL(
                """
                SELECT * FROM {}
                WHERE work_kind IN (
                    'provider_finish',
                    'provider_failure_report',
                    'provider_cancellation_report'
                )
                  AND fencing_token IS NULL
                  AND next_attempt_at <= %s
                  AND (
                    (work_kind = 'provider_finish' AND deadline_at > %s)
                    OR (
                      work_kind IN (
                        'provider_failure_report',
                        'provider_cancellation_report'
                      )
                      AND COALESCE(terminal_report_deadline_at, deadline_at) > %s
                    )
                  )
                ORDER BY next_attempt_at, created_at
                FOR UPDATE SKIP LOCKED
                LIMIT 1
                """
            ).format(self._table),
            (now, now, now),
        )
        return await cursor.fetchone()

    async def _select_processing(
        self,
        connection: AsyncConnection[DictRow],
        now: datetime,
    ) -> DictRow | None:
        workflows_cursor = await connection.execute(
            sql.SQL(
                """
                SELECT workflow_id, min(created_at) AS first_created
                FROM {}
                WHERE work_kind = 'processing'
                  AND fencing_token IS NULL
                  AND next_attempt_at <= %s
                  AND deadline_at > %s
                GROUP BY workflow_id
                ORDER BY first_created
                """
            ).format(self._table),
            (now, now),
        )
        for workflow in await workflows_cursor.fetchall():
            workflow_id = cast("str", workflow["workflow_id"])
            lock_cursor = await connection.execute(
                "SELECT pg_try_advisory_xact_lock(hashtextextended(%s, 0)) AS locked",
                (workflow_id,),
            )
            lock_row = await lock_cursor.fetchone()
            if lock_row is None or not lock_row["locked"]:
                continue
            active_cursor = await connection.execute(
                sql.SQL(
                    """
                    SELECT count(*) AS active FROM {}
                    WHERE workflow_id = %s
                      AND work_kind = 'processing'
                      AND fencing_token IS NOT NULL
                      AND lease_expires_at > %s
                    """
                ).format(self._table),
                (workflow_id, now),
            )
            active_row = await active_cursor.fetchone()
            assert active_row is not None
            if active_row["active"] >= self._per_workflow_limit:
                continue
            candidate_cursor = await connection.execute(
                sql.SQL(
                    """
                    SELECT * FROM {}
                    WHERE workflow_id = %s
                      AND work_kind = 'processing'
                      AND fencing_token IS NULL
                      AND next_attempt_at <= %s
                      AND deadline_at > %s
                    ORDER BY next_attempt_at, created_at
                    FOR UPDATE SKIP LOCKED
                    LIMIT 1
                    """
                ).format(self._table),
                (workflow_id, now, now),
            )
            candidate = await candidate_cursor.fetchone()
            if candidate is not None:
                return candidate
        return None

    async def _recover_expired_leases(
        self,
        connection: AsyncConnection[DictRow],
        now: datetime,
    ) -> None:
        cursor = await connection.execute(
            sql.SQL(
                """
                SELECT * FROM {}
                WHERE fencing_token IS NOT NULL AND lease_expires_at <= %s
                FOR UPDATE SKIP LOCKED
                """
            ).format(self._table),
            (now,),
        )
        for row in await cursor.fetchall():
            execution = self._record(row)
            recovered = self._transitions.recover_expired(execution, now)
            await self._persist_state(connection, recovered)

    async def _expire_deadlines(
        self,
        connection: AsyncConnection[DictRow],
        now: datetime,
    ) -> None:
        cursor = await connection.execute(
            sql.SQL(
                """
                SELECT * FROM {}
                WHERE (
                    status NOT IN ('succeeded', 'failed', 'cancelled')
                    AND deadline_at <= %s
                  ) OR (
                    status IN ('failed', 'cancelled')
                    AND work_kind IN (
                      'provider_failure_report',
                      'provider_cancellation_report'
                    )
                    AND COALESCE(terminal_report_deadline_at, deadline_at) <= %s
                  )
                FOR UPDATE SKIP LOCKED
                """
            ).format(self._table),
            (now, now),
        )
        for row in await cursor.fetchall():
            execution = self._record(row)
            await self._persist_state(
                connection,
                self._transitions.deadline(execution, now),
            )

    async def _persist_state(
        self,
        connection: AsyncConnection[DictRow],
        execution: ExecutionRecord,
    ) -> ExecutionRecord:
        cursor = await connection.execute(
            sql.SQL(
                """
                UPDATE {} SET
                    status = %s, work_kind = %s, next_attempt_at = %s,
                    processing_attempts = %s, processing_rejections = %s,
                    finish_attempts = %s, finish_mutation_started = %s,
                    lease_owner = %s,
                    fencing_token = %s, lease_expires_at = %s,
                    result = %s, error_code = %s, error_message = %s,
                    updated_at = %s
                WHERE id = %s
                RETURNING *
                """
            ).format(self._table),
            (
                execution.status,
                execution.work_kind.value if execution.work_kind is not None else None,
                execution.next_attempt_at,
                execution.processing_attempts,
                execution.processing_rejections,
                execution.finish_attempts,
                execution.finish_mutation_started,
                execution.lease_owner,
                execution.fencing_token,
                execution.lease_expires_at,
                Jsonb(result_to_dict(execution.result))
                if execution.result is not None
                else None,
                execution.error.code if execution.error is not None else None,
                execution.error.message if execution.error is not None else None,
                execution.updated_at,
                execution.id,
            ),
        )
        row = await cursor.fetchone()
        assert row is not None
        return self._record(row)

    async def _fetch(
        self,
        connection: AsyncConnection[DictRow],
        execution_id: str,
        *,
        for_update: bool = False,
    ) -> DictRow:
        suffix = sql.SQL(" FOR UPDATE") if for_update else sql.SQL("")
        cursor = await connection.execute(
            sql.SQL("SELECT * FROM {} WHERE id = %s").format(self._table) + suffix,
            (execution_id,),
        )
        row = await cursor.fetchone()
        if row is None:
            raise UnknownExecutionError(execution_id)
        return row

    async def _database_now(self, connection: AsyncConnection[DictRow]) -> datetime:
        cursor = await connection.execute("SELECT clock_timestamp() AS now")
        row = await cursor.fetchone()
        assert row is not None
        return cast("datetime", row["now"])

    @asynccontextmanager
    async def _connection(self) -> AsyncIterator[AsyncConnection[DictRow]]:
        try:
            async with await AsyncConnection.connect(
                self._dsn,
                row_factory=dict_row,
            ) as connection:
                yield connection
        except PsycopgError as error:
            raise ExecutionRepositoryUnavailableError(
                "execution repository is unavailable"
            ) from error

    def _record(self, row: DictRow) -> ExecutionRecord:
        error = None
        if row["error_code"] is not None:
            error = ExecutionError(row["error_code"], row["error_message"])
        return ExecutionRecord(
            id=str(row["id"]),
            identity=SubmissionIdentity(
                provider=row["provider"],
                key=row["submission_key"],
                fingerprint=row["request_fingerprint"],
            ),
            workflow_id=row["workflow_id"],
            status=ExecutionStatus(row["status"]),
            invocation=invocation_from_dict(row["invocation"]),
            protected_context=self._cipher.decrypt(bytes(row["protected_context"])),
            deadline_at=row["deadline_at"],
            terminal_report_deadline_at=row["terminal_report_deadline_at"],
            work_kind=WorkKind(row["work_kind"]) if row["work_kind"] else None,
            next_attempt_at=row["next_attempt_at"],
            processing_attempts=row["processing_attempts"],
            processing_rejections=row["processing_rejections"],
            finish_attempts=row["finish_attempts"],
            finish_mutation_started=row["finish_mutation_started"],
            lease_owner=row["lease_owner"],
            fencing_token=(
                str(row["fencing_token"]) if row["fencing_token"] is not None else None
            ),
            lease_expires_at=row["lease_expires_at"],
            result=result_from_dict(row["result"]) if row["result"] else None,
            error=error,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _check_fingerprint(
        execution: ExecutionRecord,
        identity: SubmissionIdentity,
    ) -> None:
        if execution.identity.fingerprint != identity.fingerprint:
            raise ScheduleConflictError(
                "submission identity was reused with a different request"
            )
