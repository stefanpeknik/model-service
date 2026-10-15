"""Apply versioned Yoyo migrations without blocking the scheduler's event loop."""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from urllib.parse import urlencode

from psycopg import Connection, Error, sql
from psycopg.conninfo import conninfo_to_dict
from yoyo import get_backend, read_migrations
from yoyo.exceptions import BadMigration, LockTimeout, MigrationConflict

from model_service.scheduler.store.models import DEFAULT_SCHEMA
from model_service.scheduler.store.repository import ExecutionRepositoryUnavailableError


MIGRATIONS_DIRECTORY = Path(__file__).with_name("migrations")


async def initialize_execution_schema(dsn: str, *, schema: str) -> None:
    """Apply pending migrations in a worker thread, closing all database connections."""
    try:
        await asyncio.to_thread(_apply_migrations, dsn, schema)
    except (Error, BadMigration, LockTimeout, MigrationConflict) as error:
        raise ExecutionRepositoryUnavailableError(
            "execution schema migrations could not be applied"
        ) from error


def _apply_migrations(dsn: str, schema: str) -> None:
    """Bootstrap the target schema and apply migrations under initialization locks."""
    migrations = read_migrations(str(MIGRATIONS_DIRECTORY))
    if not migrations:
        raise BadMigration("execution migration directory is empty")
    with Connection.connect(dsn, autocommit=True) as connection:
        # Serialize schema creation and Yoyo's own bootstrap, not just migration steps.
        connection.execute(
            "SELECT pg_advisory_lock(hashtextextended(%s, 0))",
            (f"model-service-schema:{schema}",),
        )
        connection.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema))
        )
        parameters = conninfo_to_dict(dsn)
        parameters["schema"] = sql.Identifier(schema).as_string(connection)
        uri = "postgresql+psycopg:///?" + urlencode(parameters)
        with get_backend(uri) as backend, backend.lock():
            backend.apply_migrations(backend.to_apply(migrations))
            if backend.to_apply(migrations):
                raise BadMigration("execution migrations remain unapplied")


def main() -> None:
    """Apply migrations using DATABASE_URL without putting credentials in CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", default=DEFAULT_SCHEMA)
    args = parser.parse_args()
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        parser.error("set DATABASE_URL to the target PostgreSQL database")
    try:
        asyncio.run(initialize_execution_schema(dsn, schema=args.schema))
    except ExecutionRepositoryUnavailableError:
        parser.exit(1, "execution schema migrations could not be applied\n")


if __name__ == "__main__":
    main()
