"""Check real PostgreSQL admission races, locking, leases, schema compatibility, and encrypted context.

Requires TEST_POSTGRES_DSN. Each case creates and later drops its own randomly named
schema; no deployed scheduler or Ray service is used.
"""

from __future__ import annotations

import asyncio
import base64
import os
import shutil
import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4

from psycopg import AsyncConnection, sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.errors import CheckViolation
from psycopg.types.json import Jsonb

from execution_scheduler.contracts import ScheduleConflictError
from execution_scheduler.domain import (
    Admission,
    ProcessingSucceeded,
    ProviderFinishSucceeded,
    RenewDirective,
    RetryPolicy,
    StaleWorkClaimError,
    SubmissionIdentity,
    WorkKind,
)
from execution_scheduler.persistence.models import ExecutionRow
from execution_scheduler.persistence.postgres import PostgresExecutionRepository
from execution_scheduler.persistence.protected_context import ProtectedContextCipher
from execution_scheduler.persistence.schema import MIGRATIONS_DIRECTORY
from execution_scheduler.repository import ExecutionRepositoryUnavailableError
from tests.repository_contract import ExecutionRepositoryContract
from workflow_execution.contracts import (
    InlineInput,
    InlineOutput,
    InputAccessContext,
    SharedArtifact,
    SharedPathInput,
    WorkflowInvocation,
    WorkflowResult,
)
from workflow_execution.serialization import invocation_to_dict, result_to_dict


DATABASE_DEADLINE_WINDOW = timedelta(milliseconds=500)
DATABASE_DEADLINE_WAIT_SECONDS = 0.7


class PostgresExecutionRepositoryTests(
    ExecutionRepositoryContract, unittest.IsolatedAsyncioTestCase
):
    """Exercise real PostgreSQL admission races, locking, leases, schema compatibility, and encrypted context."""

    async def _advance_time(self, duration: timedelta) -> None:
        """Advance the repository test clock or wait for database time according to the concrete adapter.

        Args:
            duration: Amount of time to advance or wait in a deterministic lifecycle
                scenario.

        Returns:
            None: No return value.
        """
        await asyncio.sleep(duration.total_seconds())

    async def asyncSetUp(self) -> None:
        """Create isolated fixtures and clients for real PostgreSQL admission races, locking, leases, schema compatibility, and encrypted context.

        Returns:
            None: No return value.
        """
        dsn = os.environ.get("TEST_POSTGRES_DSN")
        if dsn is None:
            self.skipTest("TEST_POSTGRES_DSN is not configured")
        self.dsn = dsn
        self.schema = f"test_execution_{uuid4().hex}"
        key = base64.urlsafe_b64encode(bytes(range(32))).decode()
        self.cipher = ProtectedContextCipher.from_base64(key)
        self.processing_policy = RetryPolicy(
            max_attempts=3,
            initial_backoff=timedelta(0),
            maximum_backoff=timedelta(0),
        )
        self.repository = self._repository()
        await self.repository.initialize()

    async def asyncTearDown(self) -> None:
        """Release test-owned clients and isolated storage after the case completes.

        Returns:
            None: No return value.
        """
        if not hasattr(self, "dsn"):
            return
        async with await AsyncConnection.connect(self.dsn) as connection:
            await connection.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(self.schema))
            )

    async def test_concurrent_equivalent_admission_creates_one_execution(self) -> None:
        """Verify concurrent equivalent admission creates one execution.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        admission = self._admission("submission-1")

        first, second = await asyncio.gather(
            self.repository.admit(admission),
            self._repository().admit(admission),
        )

        self.assertEqual({first.created, second.created}, {False, True})
        self.assertEqual(first.execution.id, second.execution.id)

    async def test_readiness_checks_initialized_execution_storage(self) -> None:
        """Verify readiness checks initialized execution storage.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        await self.repository.check_ready()

    async def test_initialize_is_safe_for_concurrent_repositories(self) -> None:
        """Verify initialize is safe for concurrent repositories.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        admitted = await self.repository.admit(self._admission("existing"))

        await asyncio.gather(*(self._repository().initialize() for _ in range(4)))

        self.assertEqual(
            await self.repository.get(admitted.execution.id), admitted.execution
        )

    async def test_untracked_legacy_database_is_rejected_without_losing_data(
        self,
    ) -> None:
        execution_id, _, _ = await self._install_pre_orm_record()
        before = await self._stored_record(execution_id)

        with self.assertRaises(ExecutionRepositoryUnavailableError):
            await self.repository.initialize()

        self.assertEqual(await self._stored_record(execution_id), before)
        self.assertEqual(await self._applied_migrations(), set())

    async def test_initial_migration_matches_orm_columns_and_indexes(self) -> None:
        async with await AsyncConnection.connect(self.dsn) as connection:
            columns = await connection.execute(
                "SELECT column_name, is_nullable FROM information_schema.columns "
                "WHERE table_schema = %s AND table_name = 'executions'",
                (self.schema,),
            )
            actual_columns = dict(await columns.fetchall())
            indexes = await connection.execute(
                "SELECT indexname FROM pg_indexes WHERE schemaname = %s",
                (self.schema,),
            )
            index_names = {row[0] for row in await indexes.fetchall()}
        self.assertEqual(
            actual_columns,
            {
                column.name: "YES" if column.nullable else "NO"
                for column in ExecutionRow.__table__.columns
            },
        )
        self.assertTrue(
            {"executions_due_idx", "executions_active_processing_idx"} <= index_names
        )
        self.assertEqual(await self._applied_migrations(), {"0001-create-executions"})

    async def test_concurrent_initialization_of_fresh_schema(self) -> None:
        other_schema = f"test_execution_{uuid4().hex}"
        self.addAsyncCleanup(self._drop_schema, other_schema)
        repositories = [self._repository(schema=other_schema) for _ in range(4)]
        await asyncio.gather(*(repository.initialize() for repository in repositories))
        await repositories[0].check_ready()
        admitted = await repositories[0].admit(self._admission("fresh-schema"))
        self.assertEqual(
            await repositories[1].get(admitted.execution.id), admitted.execution
        )

    async def test_pending_migration_runs_once_without_changing_records(self) -> None:
        admitted = await self.repository.admit(self._admission("before-upgrade"))
        before = await self._stored_record(admitted.execution.id)
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            shutil.copy(MIGRATIONS_DIRECTORY / "0001-create-executions.sql", directory)
            (directory / "0002-test-upgrade.sql").write_text(
                "-- depends: 0001-create-executions\n"
                "ALTER TABLE executions ADD COLUMN migration_test_marker TEXT;\n",
                encoding="utf-8",
            )
            with patch(
                "execution_scheduler.persistence.schema.MIGRATIONS_DIRECTORY", directory
            ):
                await asyncio.gather(
                    self.repository.initialize(), self._repository().initialize()
                )
                await self.repository.initialize()

        after = await self._stored_record(admitted.execution.id)
        self.assertIsNone(after.pop("migration_test_marker"))
        self.assertEqual(after, before)
        self.assertEqual(
            await self._applied_migrations(),
            {"0001-create-executions", "0002-test-upgrade"},
        )
        self.assertEqual(
            await self.repository.get(admitted.execution.id), admitted.execution
        )

    async def test_failed_migration_rolls_back_and_can_be_retried(self) -> None:
        admitted = await self.repository.admit(self._admission("before-failure"))
        before = await self._stored_record(admitted.execution.id)
        with TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            shutil.copy(MIGRATIONS_DIRECTORY / "0001-create-executions.sql", directory)
            migration = directory / "0002-test-upgrade.sql"
            migration.write_text(
                "-- depends: 0001-create-executions\n"
                "ALTER TABLE executions ADD COLUMN migration_test_marker TEXT;\n"
                "SELECT 1 / 0;\n",
                encoding="utf-8",
            )
            with patch(
                "execution_scheduler.persistence.schema.MIGRATIONS_DIRECTORY", directory
            ):
                with self.assertRaises(ExecutionRepositoryUnavailableError):
                    await self.repository.initialize()
                self.assertEqual(
                    await self._stored_record(admitted.execution.id), before
                )
                self.assertEqual(
                    await self._applied_migrations(), {"0001-create-executions"}
                )
                migration.write_text(
                    "-- depends: 0001-create-executions\n"
                    "ALTER TABLE executions ADD COLUMN migration_test_marker TEXT;\n",
                    encoding="utf-8",
                )
                await self.repository.initialize()

        self.assertEqual(
            await self._applied_migrations(),
            {"0001-create-executions", "0002-test-upgrade"},
        )
        self.assertEqual(
            await self.repository.get(admitted.execution.id), admitted.execution
        )

    async def test_empty_migration_directory_fails_initialization(self) -> None:
        admitted = await self.repository.admit(self._admission("missing-migrations"))
        with (
            TemporaryDirectory() as temporary_directory,
            patch(
                "execution_scheduler.persistence.schema.MIGRATIONS_DIRECTORY",
                Path(temporary_directory),
            ),
            self.assertRaises(ExecutionRepositoryUnavailableError),
        ):
            await self.repository.initialize()
        self.assertEqual(
            await self.repository.get(admitted.execution.id), admitted.execution
        )

    async def _applied_migrations(self) -> set[str]:
        async with await AsyncConnection.connect(self.dsn) as connection:
            cursor = await connection.execute(
                sql.SQL("SELECT migration_id FROM {}").format(
                    sql.Identifier(self.schema, "_yoyo_migration")
                )
            )
            return {row[0] for row in await cursor.fetchall()}

    async def test_libpq_connection_string_remains_supported(self) -> None:
        """Verify libpq connection string remains supported.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        repository = PostgresExecutionRepository(
            make_conninfo(**conninfo_to_dict(self.dsn)),
            cipher=self.cipher,
            schema=self.schema,
        )

        await repository.initialize()
        admitted = await repository.admit(self._admission("libpq"))

        self.assertEqual(
            await self.repository.get(admitted.execution.id), admitted.execution
        )

    async def test_schema_translation_keeps_repositories_isolated(self) -> None:
        """Verify schema translation keeps repositories isolated.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        other_schema = f"test_execution_{uuid4().hex}"
        self.addAsyncCleanup(self._drop_schema, other_schema)
        other = self._repository(schema=other_schema)
        await other.initialize()
        admission = self._admission("same-key")

        first = await self.repository.admit(admission)
        second = await other.admit(admission)

        self.assertTrue(first.created)
        self.assertTrue(second.created)
        self.assertNotEqual(first.execution.id, second.execution.id)
        first_found = await self.repository.find_submission(admission.identity)
        second_found = await other.find_submission(admission.identity)
        assert first_found is not None
        assert second_found is not None
        self.assertEqual(first_found.id, first.execution.id)
        self.assertEqual(second_found.id, second.execution.id)

    async def test_parallel_claims_respect_capacity_greater_than_one(self) -> None:
        """Verify parallel claims respect capacity greater than one.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        repository = self._repository(per_workflow_limit=2)
        for index in range(6):
            await repository.admit(self._admission(f"capacity-{index}"))
        claims = []

        for _ in range(3):
            competing = await asyncio.gather(
                *(
                    self._repository(per_workflow_limit=2).claim_due(
                        f"worker-{index}", timedelta(seconds=30)
                    )
                    for index in range(6)
                )
            )
            claims.extend(claim for claim in competing if claim is not None)

        self.assertEqual(len(claims), 2)
        self.assertEqual(len({claim.execution_id for claim in claims}), 2)
        self.assertIsNone(
            await repository.claim_due("extra-worker", timedelta(seconds=30))
        )

    async def test_locked_execution_does_not_block_another_workflow(self) -> None:
        """Verify locked execution does not block another workflow.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        locked_id = await self._admit_execution("locked", "first-workflow")
        available_id = await self._admit_execution("available", "second-workflow")

        async with await AsyncConnection.connect(self.dsn) as connection:
            await connection.execute(
                sql.SQL("SELECT id FROM {} WHERE id = %s FOR UPDATE").format(
                    sql.Identifier(self.schema, "executions")
                ),
                (locked_id,),
            )
            claim = await asyncio.wait_for(
                self.repository.claim_due("worker", timedelta(seconds=30)), timeout=2
            )

        assert claim is not None
        self.assertEqual(claim.execution_id, available_id)
        self.assertEqual((await self.repository.get(locked_id)).processing_attempts, 0)

    async def test_failed_claim_rolls_back_and_releases_workflow_lock(self) -> None:
        """Verify failed claim rolls back and releases workflow lock.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        admitted = await self.repository.admit(self._admission("rollback"))
        table = sql.Identifier(self.schema, "executions")
        async with await AsyncConnection.connect(self.dsn) as connection:
            await connection.execute(
                sql.SQL(
                    "ALTER TABLE {} ADD CONSTRAINT reject_running CHECK (status <> 'running')"
                ).format(table)
            )

        with self.assertRaises(ExecutionRepositoryUnavailableError):
            await self.repository.claim_due("rejected-worker", timedelta(seconds=30))

        self.assertEqual(
            await self.repository.get(admitted.execution.id), admitted.execution
        )
        async with await AsyncConnection.connect(self.dsn) as connection:
            await connection.execute(
                sql.SQL("ALTER TABLE {} DROP CONSTRAINT reject_running").format(table)
            )
        claim = await self._repository().claim_due("replacement", timedelta(seconds=30))
        assert claim is not None
        self.assertEqual(claim.execution_id, admitted.execution.id)
        self.assertEqual(claim.attempt, 1)

    async def test_orm_schema_retains_database_state_constraints(self) -> None:
        """Verify ORM schema retains database state constraints.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        admitted = await self.repository.admit(self._admission("constraints"))
        for assignment in (
            "processing_attempts = -1",
            "processing_rejections = -1",
            "finish_attempts = -1",
            "lease_owner = 'worker-without-token'",
            "error_code = 'code-without-message'",
            "result = '{}'::jsonb",
            "status = 'invalid-status'",
            "work_kind = 'invalid-work-kind'",
        ):
            with self.subTest(assignment=assignment), self.assertRaises(CheckViolation):
                async with await AsyncConnection.connect(self.dsn) as connection:
                    await connection.execute(
                        sql.SQL("UPDATE {} SET {} WHERE id = %s").format(
                            sql.Identifier(self.schema, "executions"),
                            sql.SQL(assignment),
                        ),
                        (admitted.execution.id,),
                    )
        self.assertEqual(
            await self.repository.get(admitted.execution.id), admitted.execution
        )

    async def _drop_schema(self, schema: str) -> None:
        """Drop only the explicitly named isolated PostgreSQL test schema with its contained objects.

        Args:
            schema: Validated PostgreSQL schema name, or a separate isolated schema for
                the test.

        Returns:
            None: No return value.
        """
        async with await AsyncConnection.connect(self.dsn) as connection:
            await connection.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema))
            )

    async def _install_pre_orm_record(self) -> tuple[str, Admission, WorkflowResult]:
        """Install historical SQL schema/data with encrypted context and stored outputs to verify ORM initialization preserves them.

        Returns:
            tuple[str, Admission, WorkflowResult]: Selected records, identifiers, or
                fixture values in the ordering described above.
        """
        admission = self._admission("pre-orm")
        admission = replace(
            admission,
            invocation=replace(
                admission.invocation,
                inputs={
                    "slide": SharedPathInput(path="/shared/input.tiff"),
                    "threshold": InlineInput(value=0.5),
                },
                resolved_configuration={"level": 2, "optional": None},
            ),
            protected_context={"credential": "historical-test-token"},
        )
        result = WorkflowResult(
            outputs={
                "score": InlineOutput(value=0.9),
                "empty": InlineOutput(value={}),
                "artifact": SharedArtifact(
                    path="/shared/result.tiff", media_type="image/tiff"
                ),
            }
        )
        execution_id = str(uuid4())
        table = sql.Identifier(self.schema, "executions")
        fixture = Path(__file__).with_name("fixtures") / "executions_pre_orm.sql"
        async with await AsyncConnection.connect(self.dsn) as connection:
            await connection.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(self.schema))
            )
            await connection.execute(
                sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(self.schema))
            )
            await connection.execute(sql.SQL(fixture.read_text()).format(table=table))
            await connection.execute(
                sql.SQL(
                    "INSERT INTO {} (id, provider, submission_key, request_fingerprint, "
                    "workflow_id, status, invocation, protected_context, deadline_at, "
                    "work_kind, next_attempt_at, processing_attempts, result) "
                    "VALUES (%s, %s, %s, %s, %s, 'running', %s, %s, %s, "
                    "'provider_finish', clock_timestamp(), 1, %s)"
                ).format(table),
                (
                    execution_id,
                    admission.identity.provider,
                    admission.identity.key,
                    admission.identity.fingerprint,
                    admission.workflow_id,
                    Jsonb(invocation_to_dict(admission.invocation)),
                    self.cipher.encrypt(admission.protected_context),
                    admission.deadline_at,
                    Jsonb(result_to_dict(result)),
                ),
            )
        return execution_id, admission, result

    async def _stored_record(self, execution_id: str) -> dict[str, object]:
        """Read the raw PostgreSQL execution row as JSON for persistence and compatibility assertions.

        Args:
            execution_id: Durable execution identifier, not an external provider job ID.

        Returns:
            dict[str, object]: Decoded, serialized, or recorded fields described above.
        """
        async with await AsyncConnection.connect(self.dsn) as connection:
            cursor = await connection.execute(
                sql.SQL("SELECT row_to_json(executions) FROM {} WHERE id = %s").format(
                    sql.Identifier(self.schema, "executions")
                ),
                (execution_id,),
            )
            row = await cursor.fetchone()
        assert row is not None
        return row[0]

    async def test_maintenance_expires_queued_work_without_claiming(self) -> None:
        """Verify maintenance expires queued work without claiming.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        admission = self._admission("queued-expiry")
        admitted = await self.repository.admit(
            Admission(
                identity=admission.identity,
                workflow_id=admission.workflow_id,
                invocation=admission.invocation,
                protected_context=admission.protected_context,
                deadline_at=datetime.now(UTC) + DATABASE_DEADLINE_WINDOW,
            )
        )
        await asyncio.sleep(DATABASE_DEADLINE_WAIT_SECONDS)

        await self._repository().maintain()

        failed = await self.repository.get(admitted.execution.id)
        self.assertEqual(failed.status, "failed")
        self.assertIsNone(failed.fencing_token)
        self.assertEqual(failed.processing_attempts, 0)
        assert failed.error is not None
        self.assertEqual(failed.error.code, "execution_deadline_exceeded")

    async def test_maintenance_recovers_expired_lease_without_claiming(self) -> None:
        """Verify maintenance recovers expired lease without claiming.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        admitted = await self.repository.admit(self._admission("lease-recovery"))
        claim = await self.repository.claim_due("worker-1", timedelta(milliseconds=30))
        assert claim is not None
        await asyncio.sleep(0.08)

        await self._repository().maintain()

        recovered = await self.repository.get(admitted.execution.id)
        self.assertEqual(recovered.status, "queued")
        self.assertIsNone(recovered.fencing_token)
        self.assertEqual(recovered.processing_attempts, 1)

    async def test_expired_lease_rejects_settlement_before_recovery(self) -> None:
        """Verify expired lease rejects settlement before recovery.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        admitted = await self.repository.admit(self._admission("late-result"))
        claim = await self.repository.claim_due("worker-1", timedelta(milliseconds=30))
        assert claim is not None
        await asyncio.sleep(0.08)

        with self.assertRaises(StaleWorkClaimError):
            await self.repository.settle(
                claim.execution_id,
                claim.fencing_token,
                ProcessingSucceeded(WorkflowResult(outputs={})),
            )
        durable = await self.repository.get(admitted.execution.id)
        self.assertIsNone(durable.result)
        self.assertEqual(durable.fencing_token, claim.fencing_token)

    async def test_conflicting_admission_is_rejected(self) -> None:
        """Verify conflicting admission is rejected.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        first = self._admission("submission-1")
        conflicting = Admission(
            identity=SubmissionIdentity(
                provider=first.identity.provider,
                key=first.identity.key,
                fingerprint="different-request",
            ),
            workflow_id=first.workflow_id,
            invocation=first.invocation,
            protected_context=first.protected_context,
            deadline_at=first.deadline_at,
        )
        await self.repository.admit(first)

        with self.assertRaises(ScheduleConflictError):
            await self._repository().admit(conflicting)

    async def test_workflow_limit_and_fencing_hold_across_connections(self) -> None:
        """Verify workflow limit and fencing hold across connections.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        first = await self.repository.admit(self._admission("submission-1"))
        second = await self.repository.admit(self._admission("submission-2"))

        first_claim, competing_claim = await asyncio.gather(
            self.repository.claim_due("worker-1", timedelta(seconds=30)),
            self._repository().claim_due("worker-2", timedelta(seconds=30)),
        )

        claims = [
            claim for claim in (first_claim, competing_claim) if claim is not None
        ]
        self.assertEqual(len(claims), 1)
        claimed = claims[0]
        self.assertIn(
            claimed.execution_id,
            {first.execution.id, second.execution.id},
        )

    async def test_expired_processing_lease_is_recovered_and_stale_settlement_fails(
        self,
    ) -> None:
        """Verify expired processing lease is recovered and stale settlement fails.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        admitted = await self.repository.admit(self._admission("submission-1"))
        abandoned = await self.repository.claim_due(
            "worker-1",
            timedelta(milliseconds=30),
        )
        assert abandoned is not None

        await asyncio.sleep(0.08)
        replacement = await self._repository().claim_due(
            "worker-2",
            timedelta(seconds=30),
        )

        assert replacement is not None
        self.assertEqual(replacement.execution_id, admitted.execution.id)
        self.assertNotEqual(replacement.fencing_token, abandoned.fencing_token)
        self.assertEqual(replacement.attempt, 2)
        result = WorkflowResult(outputs={"score": InlineOutput(value=0.9)})
        with self.assertRaises(StaleWorkClaimError):
            await self.repository.settle(
                abandoned.execution_id,
                abandoned.fencing_token,
                ProcessingSucceeded(result),
            )
        processing_complete = await self.repository.settle(
            replacement.execution_id,
            replacement.fencing_token,
            ProcessingSucceeded(result),
        )
        self.assertEqual(processing_complete.status, "running")
        self.assertEqual(processing_complete.work_kind, WorkKind.PROVIDER_FINISH)
        self.assertEqual(processing_complete.result, result)

        finish = await self.repository.claim_due(
            "worker-2",
            timedelta(seconds=30),
        )
        assert finish is not None
        succeeded = await self.repository.settle(
            finish.execution_id,
            finish.fencing_token,
            ProviderFinishSucceeded(),
        )
        self.assertEqual(succeeded.status, "succeeded")
        self.assertEqual(succeeded.result, result)

    async def test_renewal_and_cancellation_fence_competing_workers(self) -> None:
        """Verify renewal and cancellation fence competing workers.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        admitted = await self.repository.admit(self._admission("submission-1"))
        claim = await self.repository.claim_due("worker-1", timedelta(seconds=1))
        assert claim is not None

        directive = await self.repository.renew(
            claim.execution_id,
            claim.fencing_token,
            timedelta(seconds=1),
        )
        competing = await self._repository().claim_due(
            "worker-2",
            timedelta(seconds=1),
        )

        self.assertEqual(directive, RenewDirective.CONTINUE)
        self.assertIsNone(competing)
        cancelled = await self._repository().request_cancel(admitted.execution.id)
        self.assertEqual(cancelled.status, "cancelled")
        self.assertEqual(
            await self.repository.renew(
                claim.execution_id,
                claim.fencing_token,
                timedelta(seconds=1),
            ),
            RenewDirective.CANCEL,
        )
        with self.assertRaises(StaleWorkClaimError):
            await self.repository.settle(
                claim.execution_id,
                claim.fencing_token,
                ProcessingSucceeded(WorkflowResult(outputs={})),
            )

    async def test_expired_finish_lease_preserves_result_and_processing_attempt(
        self,
    ) -> None:
        """Verify expired finish lease preserves result and processing attempt.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        admitted = await self.repository.admit(self._admission("submission-1"))
        processing = await self.repository.claim_due(
            "worker-1",
            timedelta(seconds=1),
        )
        assert processing is not None
        result = WorkflowResult(outputs={"score": InlineOutput(value=0.9)})
        await self.repository.settle(
            processing.execution_id,
            processing.fencing_token,
            ProcessingSucceeded(result),
        )
        abandoned_finish = await self.repository.claim_due(
            "worker-1",
            timedelta(milliseconds=30),
        )
        assert abandoned_finish is not None

        await asyncio.sleep(0.08)
        replacement = await self._repository().claim_due(
            "worker-2",
            timedelta(seconds=1),
        )

        assert replacement is not None
        self.assertEqual(replacement.execution_id, admitted.execution.id)
        self.assertEqual(replacement.kind, WorkKind.PROVIDER_FINISH)
        self.assertEqual(replacement.result, result)
        self.assertEqual(replacement.attempt, 2)
        durable = await self.repository.get(admitted.execution.id)
        self.assertEqual(durable.processing_attempts, 1)

    async def test_database_deadline_rejects_a_late_settlement(self) -> None:
        """Verify database deadline rejects a late settlement.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        admission = self._admission("submission-1")
        expiring = Admission(
            identity=admission.identity,
            workflow_id=admission.workflow_id,
            invocation=admission.invocation,
            protected_context=admission.protected_context,
            deadline_at=datetime.now(UTC) + DATABASE_DEADLINE_WINDOW,
        )
        admitted = await self.repository.admit(expiring)
        claim = await self.repository.claim_due("worker-1", timedelta(seconds=1))
        assert claim is not None

        await asyncio.sleep(DATABASE_DEADLINE_WAIT_SECONDS)
        with self.assertRaises(StaleWorkClaimError):
            await self.repository.settle(
                claim.execution_id,
                claim.fencing_token,
                ProcessingSucceeded(WorkflowResult(outputs={})),
            )
        failed = await self.repository.get(admitted.execution.id)
        self.assertEqual(failed.status, "failed")
        assert failed.error is not None
        self.assertEqual(failed.error.code, "execution_deadline_exceeded")

    async def test_deadline_failure_remains_reportable_until_provider_deadline(
        self,
    ) -> None:
        """Verify deadline failure remains reportable until provider deadline.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        admission = self._admission("submission-report-deadline")
        execution_deadline = datetime.now(UTC) + DATABASE_DEADLINE_WINDOW
        expiring = Admission(
            identity=admission.identity,
            workflow_id=admission.workflow_id,
            invocation=admission.invocation,
            protected_context=admission.protected_context,
            deadline_at=execution_deadline,
            terminal_report_deadline_at=datetime.now(UTC) + timedelta(minutes=5),
        )
        await self.repository.admit(expiring)
        processing = await self.repository.claim_due(
            "worker-1",
            timedelta(seconds=1),
        )
        assert processing is not None
        await asyncio.sleep(DATABASE_DEADLINE_WAIT_SECONDS)

        self.assertEqual(
            await self.repository.renew(
                processing.execution_id,
                processing.fencing_token,
                timedelta(seconds=1),
            ),
            RenewDirective.DEADLINE_EXCEEDED,
        )
        report = await self.repository.claim_due(
            "worker-2",
            timedelta(seconds=1),
        )

        assert report is not None
        self.assertEqual(report.kind, WorkKind.PROVIDER_FAILURE_REPORT)
        assert report.error is not None
        self.assertEqual(report.error.code, "execution_deadline_exceeded")

    async def test_stale_renewal_cannot_apply_deadline_transition(self) -> None:
        """Verify stale renewal cannot apply deadline transition.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        admission = self._admission("submission-1")
        expiring = Admission(
            identity=admission.identity,
            workflow_id=admission.workflow_id,
            invocation=admission.invocation,
            protected_context=admission.protected_context,
            deadline_at=datetime.now(UTC) + DATABASE_DEADLINE_WINDOW,
        )
        admitted = await self.repository.admit(expiring)
        claim = await self.repository.claim_due("worker-1", timedelta(seconds=5))
        assert claim is not None
        await asyncio.sleep(DATABASE_DEADLINE_WAIT_SECONDS)

        stale = await self._repository().renew(
            claim.execution_id,
            str(uuid4()),
            timedelta(seconds=1),
        )
        unchanged = await self.repository.get(admitted.execution.id)
        current = await self.repository.renew(
            claim.execution_id,
            claim.fencing_token,
            timedelta(seconds=1),
        )

        self.assertEqual(stale, RenewDirective.LEASE_LOST)
        self.assertEqual(unchanged.status, "running")
        self.assertEqual(current, RenewDirective.DEADLINE_EXCEEDED)
        self.assertEqual(
            (await self.repository.get(admitted.execution.id)).status,
            "failed",
        )

    async def test_protected_context_is_not_stored_as_plaintext(self) -> None:
        """Verify protected context is not stored as plaintext.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        admission = self._admission("submission-1")
        admission = Admission(
            identity=admission.identity,
            workflow_id=admission.workflow_id,
            invocation=admission.invocation,
            protected_context={"credential": "secret-job-token"},
            deadline_at=admission.deadline_at,
        )
        admitted = await self.repository.admit(admission)

        async with await AsyncConnection.connect(self.dsn) as connection:
            cursor = await connection.execute(
                sql.SQL(
                    "SELECT encode(protected_context, 'escape') FROM {} WHERE id = %s"
                ).format(sql.Identifier(self.schema, "executions")),
                (admitted.execution.id,),
            )
            row = await cursor.fetchone()

        assert row is not None
        self.assertNotIn("secret-job-token", row[0])
        loaded = await self.repository.get(admitted.execution.id)
        self.assertEqual(
            loaded.protected_context,
            {"credential": "secret-job-token"},
        )

    def _repository(
        self, *, schema: str | None = None, per_workflow_limit: int = 1
    ) -> PostgresExecutionRepository:
        """Construct an isolated PostgreSQL repository with the test cipher, schema, and zero-delay retry policy.

        Args:
            schema: Validated PostgreSQL schema name, or a separate isolated schema for
                the test.
            per_workflow_limit: Maximum simultaneous leased processing attempts per
                workflow.

        Returns:
            PostgresExecutionRepository: Repository configured for the isolated test
                schema and retry policy.
        """
        return PostgresExecutionRepository(
            self.dsn,
            cipher=self.cipher,
            schema=schema or self.schema,
            per_workflow_limit=per_workflow_limit,
            processing_policy=self.processing_policy,
            finish_policy=self.processing_policy,
        )

    @staticmethod
    def _admission(submission_key: str) -> Admission:
        """Prepare a minimal unique standalone admission with a future deadline for repository tests.

        Args:
            submission_key: Provider-scoped replay key for the repository fixture.

        Returns:
            Admission: Prepared standalone repository fixture with a future deadline.
        """
        deadline = datetime.now(UTC) + timedelta(minutes=5)
        return Admission(
            identity=SubmissionIdentity(
                provider="standalone",
                key=submission_key,
                fingerprint=f"fingerprint-{submission_key}",
            ),
            workflow_id="prostate-heatmap",
            invocation=WorkflowInvocation(
                inputs={},
                resolved_configuration={},
                input_access=InputAccessContext(
                    kind="shared_storage",
                    state={"root": "/shared"},
                ),
                deadline_at=deadline,
            ),
            protected_context={},
            deadline_at=deadline,
        )


if __name__ == "__main__":
    unittest.main()
