from __future__ import annotations

from typing import TYPE_CHECKING

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import Boolean, Column, DateTime, false, func, inspect, select
from sqlalchemy.schema import CreateSchema

from execution_scheduler.persistence.models import Base, ExecutionRow


if TYPE_CHECKING:
    from sqlalchemy import Connection
    from sqlalchemy.ext.asyncio import AsyncSession


async def initialize_execution_schema(session: AsyncSession, *, schema: str) -> None:
    await session.execute(
        select(
            func.pg_advisory_xact_lock(
                func.hashtextextended(f"model-service-schema:{schema}", 0)
            )
        )
    )
    connection = await session.connection()
    await connection.execute(CreateSchema(schema, if_not_exists=True))
    await connection.run_sync(_initialize_tables, schema)


def _initialize_tables(connection: Connection, schema: str) -> None:
    inspector = inspect(connection)
    if inspector.has_table(ExecutionRow.__tablename__, schema=schema):
        columns = {
            column["name"]
            for column in inspector.get_columns(
                ExecutionRow.__tablename__, schema=schema
            )
        }
        # Preserve the two additive upgrades supported by the pre-ORM repository.
        operations = Operations(MigrationContext.configure(connection))
        for column in (
            Column("terminal_report_deadline_at", DateTime(timezone=True)),
            Column(
                "finish_mutation_started",
                Boolean,
                nullable=False,
                server_default=false(),
            ),
        ):
            if column.name not in columns:
                operations.add_column(ExecutionRow.__tablename__, column, schema=schema)
    Base.metadata.create_all(connection)
    for table in Base.metadata.sorted_tables:
        for index in table.indexes:
            index.create(connection, checkfirst=True)
