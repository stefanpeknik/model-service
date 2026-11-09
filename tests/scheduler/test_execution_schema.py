"""Check the execution schema migrations against a real PostgreSQL.

Requires TEST_POSTGRES_DSN. Each case creates and later drops its own randomly named
schema.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4

from psycopg import AsyncConnection, sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg.errors import CheckViolation

from model_service.scheduler.store.models import ExecutionRow
from model_service.scheduler.store.repository import ExecutionRepositoryUnavailableError
from model_service.scheduler.store.schema import (
    MIGRATIONS_DIRECTORY,
    initialize_execution_schema,
)


ALL_MIGRATIONS = {path.stem for path in MIGRATIONS_DIRECTORY.glob("*.sql")}
LATEST_MIGRATION = max(ALL_MIGRATIONS)


class ExecutionSchemaTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        dsn = os.environ.get("TEST_POSTGRES_DSN")
        if dsn is None:
            self.skipTest("TEST_POSTGRES_DSN is not configured")
        self.dsn = dsn
        self.schema = f"test_execution_{uuid4().hex}"
        self.addAsyncCleanup(self._drop_schema, self.schema)
        await initialize_execution_schema(self.dsn, schema=self.schema)

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
            {
                "executions_due_idx",
                "executions_active_processing_idx",
                "executions_deadline_idx",
                "executions_report_deadline_idx",
                "executions_lease_idx",
                "executions_finished_idx",
            }
            <= index_names
        )
        self.assertEqual(await self._applied_migrations(), ALL_MIGRATIONS)

    async def test_concurrent_initialization_of_fresh_schema(self) -> None:
        other_schema = f"test_execution_{uuid4().hex}"
        self.addAsyncCleanup(self._drop_schema, other_schema)

        await asyncio.gather(
            *(
                initialize_execution_schema(self.dsn, schema=other_schema)
                for _ in range(4)
            )
        )

        self.assertEqual(await self._applied_migrations(other_schema), ALL_MIGRATIONS)

    async def test_libpq_connection_string_remains_supported(self) -> None:
        other_schema = f"test_execution_{uuid4().hex}"
        self.addAsyncCleanup(self._drop_schema, other_schema)

        await initialize_execution_schema(
            make_conninfo(**conninfo_to_dict(self.dsn)), schema=other_schema
        )

        self.assertEqual(await self._applied_migrations(other_schema), ALL_MIGRATIONS)

    async def test_pending_migration_runs_once_without_changing_records(self) -> None:
        execution_id = await self._insert_execution()
        before = await self._stored_record(execution_id)
        with TemporaryDirectory() as temporary_directory:
            directory = self._migrations_with(
                Path(temporary_directory),
                "ALTER TABLE executions ADD COLUMN migration_test_marker TEXT;\n",
            )
            with patch(
                "model_service.scheduler.store.schema.MIGRATIONS_DIRECTORY", directory
            ):
                await asyncio.gather(
                    initialize_execution_schema(self.dsn, schema=self.schema),
                    initialize_execution_schema(self.dsn, schema=self.schema),
                )
                await initialize_execution_schema(self.dsn, schema=self.schema)

        after = await self._stored_record(execution_id)
        self.assertIsNone(after.pop("migration_test_marker"))
        self.assertEqual(after, before)
        self.assertEqual(
            await self._applied_migrations(),
            ALL_MIGRATIONS | {"9999-test-upgrade"},
        )

    async def test_failed_migration_rolls_back_and_can_be_retried(self) -> None:
        execution_id = await self._insert_execution()
        before = await self._stored_record(execution_id)
        with TemporaryDirectory() as temporary_directory:
            directory = self._migrations_with(
                Path(temporary_directory),
                "ALTER TABLE executions ADD COLUMN migration_test_marker TEXT;\n"
                "SELECT 1 / 0;\n",
            )
            with patch(
                "model_service.scheduler.store.schema.MIGRATIONS_DIRECTORY", directory
            ):
                with self.assertRaises(ExecutionRepositoryUnavailableError):
                    await initialize_execution_schema(self.dsn, schema=self.schema)
                self.assertEqual(await self._stored_record(execution_id), before)
                self.assertEqual(await self._applied_migrations(), ALL_MIGRATIONS)
                self._migrations_with(
                    directory,
                    "ALTER TABLE executions ADD COLUMN migration_test_marker TEXT;\n",
                )
                await initialize_execution_schema(self.dsn, schema=self.schema)

        self.assertEqual(
            await self._applied_migrations(),
            ALL_MIGRATIONS | {"9999-test-upgrade"},
        )

    async def test_empty_migration_directory_fails_initialization(self) -> None:
        with (
            TemporaryDirectory() as temporary_directory,
            patch(
                "model_service.scheduler.store.schema.MIGRATIONS_DIRECTORY",
                Path(temporary_directory),
            ),
            self.assertRaises(ExecutionRepositoryUnavailableError),
        ):
            await initialize_execution_schema(self.dsn, schema=self.schema)

    async def test_schema_retains_database_state_constraints(self) -> None:
        execution_id = await self._insert_execution()
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
                        (execution_id,),
                    )

    @staticmethod
    def _migrations_with(directory: Path, upgrade: str) -> Path:
        for migration in MIGRATIONS_DIRECTORY.glob("*.sql"):
            shutil.copy(migration, directory)
        (directory / "9999-test-upgrade.sql").write_text(
            f"-- depends: {LATEST_MIGRATION}\n{upgrade}", encoding="utf-8"
        )
        return directory

    async def _insert_execution(self) -> str:
        execution_id = str(uuid4())
        async with await AsyncConnection.connect(self.dsn) as connection:
            await connection.execute(
                sql.SQL(
                    "INSERT INTO {} (id, provider, submission_key, request_fingerprint,"
                    " builder_id, model_id, status, invocation, protected_context,"
                    " deadline_at, work_kind, next_attempt_at)"
                    " VALUES (%s, 'standalone', 'key', 'fingerprint', 'heatmap',"
                    " 'model', 'queued', '{{}}'::jsonb, ''::bytea,"
                    " now() + interval '1 hour', 'processing', now())"
                ).format(sql.Identifier(self.schema, "executions")),
                (execution_id,),
            )
        return execution_id

    async def _stored_record(self, execution_id: str) -> dict[str, object]:
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

    async def _applied_migrations(self, schema: str | None = None) -> set[str]:
        async with await AsyncConnection.connect(self.dsn) as connection:
            cursor = await connection.execute(
                sql.SQL("SELECT migration_id FROM {}").format(
                    sql.Identifier(schema or self.schema, "_yoyo_migration")
                )
            )
            return {row[0] for row in await cursor.fetchall()}

    async def _drop_schema(self, schema: str) -> None:
        async with await AsyncConnection.connect(self.dsn) as connection:
            await connection.execute(
                sql.SQL("DROP SCHEMA IF EXISTS {} CASCADE").format(
                    sql.Identifier(schema)
                )
            )


if __name__ == "__main__":
    unittest.main()
