"""Map durable execution records and their JSON values to PostgreSQL columns.

Constraints enforce replay uniqueness and valid lease/error combinations. ORM rows stay
inside persistence; the scheduler and worker receive domain records.
"""

from __future__ import annotations

from datetime import datetime  # noqa: TC003 - SQLAlchemy resolves mapped annotations.
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    LargeBinary,
    Text,
    UniqueConstraint,
    and_,
    false,
    func,
    literal,
    or_,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import TypeDecorator

from model_service.scheduler.contracts import (
    STORED_INVOCATION,
    STORED_RESULT,
    ExecutionStatus,
    Operation,
    StoredInvocation,
    StoredResult,
)
from model_service.scheduler.domain import WorkKind


if TYPE_CHECKING:
    from sqlalchemy.engine import Dialect


DEFAULT_SCHEMA = "model_service"


class InvocationType(TypeDecorator[StoredInvocation]):
    impl = JSONB(none_as_null=True)
    cache_ok = True

    def process_bind_param(
        self, value: StoredInvocation | None, dialect: Dialect
    ) -> object:
        if value is None:
            return None
        return STORED_INVOCATION.dump_python(value, mode="json")

    def process_result_value(
        self, value: object, dialect: Dialect
    ) -> StoredInvocation | None:
        return STORED_INVOCATION.validate_python(value) if value is not None else None


class ResultType(TypeDecorator[StoredResult]):
    impl = JSONB(none_as_null=True)
    cache_ok = True

    def process_bind_param(
        self, value: StoredResult | None, dialect: Dialect
    ) -> object:
        return (
            STORED_RESULT.dump_python(value, mode="json") if value is not None else None
        )

    def process_result_value(
        self, value: object, dialect: Dialect
    ) -> StoredResult | None:
        return STORED_RESULT.validate_python(value) if value is not None else None


class Base(DeclarativeBase):
    """Collect execution ORM mappings for schema-name translation; Yoyo owns DDL."""


class ExecutionRow(Base):
    """Map one execution, including encrypted context, replay identity, pending work, counters, leases, and stored outcomes."""

    __tablename__ = "executions"
    __table_args__ = (
        UniqueConstraint("provider", "submission_key"),
        {"schema": DEFAULT_SCHEMA},
    )

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True)
    provider: Mapped[str] = mapped_column(Text)
    submission_key: Mapped[str] = mapped_column(Text)
    request_fingerprint: Mapped[str] = mapped_column(Text)
    operation: Mapped[Operation] = mapped_column(
        Enum(
            Operation,
            native_enum=False,
            create_constraint=True,
            length=None,
            values_callable=lambda enum: [operation.value for operation in enum],
        )
    )
    builder_id: Mapped[str | None] = mapped_column(Text)
    model_id: Mapped[str] = mapped_column(Text)
    status: Mapped[ExecutionStatus] = mapped_column(
        Enum(
            ExecutionStatus,
            native_enum=False,
            create_constraint=True,
            length=None,
            values_callable=lambda enum: [status.value for status in enum],
        )
    )
    invocation: Mapped[StoredInvocation] = mapped_column(InvocationType())
    protected_context: Mapped[bytes] = mapped_column(LargeBinary)
    deadline_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    terminal_report_deadline_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    work_kind: Mapped[WorkKind | None] = mapped_column(
        Enum(
            WorkKind,
            native_enum=False,
            create_constraint=True,
            length=None,
            values_callable=lambda enum: [kind.value for kind in enum],
        )
    )
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    processing_attempts: Mapped[int] = mapped_column(server_default=literal(0))
    processing_rejections: Mapped[int] = mapped_column(server_default=literal(0))
    finish_attempts: Mapped[int] = mapped_column(server_default=literal(0))
    finish_mutation_started: Mapped[bool] = mapped_column(server_default=false())
    lease_owner: Mapped[str | None] = mapped_column(Text)
    fencing_token: Mapped[str | None] = mapped_column(UUID(as_uuid=False))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    result: Mapped[StoredResult | None] = mapped_column(ResultType())
    error_code: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.clock_timestamp()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.clock_timestamp()
    )


class InferencePayloadRow(Base):
    """Request and response bytes of one inference execution."""

    __tablename__ = "inference_payloads"
    __table_args__ = ({"schema": DEFAULT_SCHEMA},)

    execution_id: Mapped[str] = mapped_column(
        UUID(as_uuid=False),
        ForeignKey(f"{DEFAULT_SCHEMA}.executions.id", ondelete="CASCADE"),
        primary_key=True,
    )
    request: Mapped[bytes | None] = mapped_column(LargeBinary)
    response: Mapped[bytes | None] = mapped_column(LargeBinary)


for counter in (
    ExecutionRow.processing_attempts,
    ExecutionRow.processing_rejections,
    ExecutionRow.finish_attempts,
):
    CheckConstraint(counter >= 0)

CheckConstraint(
    ExecutionRow.error_code.is_(None) == ExecutionRow.error_message.is_(None)
)
CheckConstraint(
    or_(
        and_(
            ExecutionRow.fencing_token.is_(None),
            ExecutionRow.lease_owner.is_(None),
            ExecutionRow.lease_expires_at.is_(None),
        ),
        and_(
            ExecutionRow.fencing_token.is_not(None),
            ExecutionRow.lease_owner.is_not(None),
            ExecutionRow.lease_expires_at.is_not(None),
        ),
    )
)
CheckConstraint(
    or_(
        ExecutionRow.result.is_(None),
        ExecutionRow.work_kind == WorkKind.PROVIDER_FINISH,
        ExecutionRow.status.in_(
            [
                ExecutionStatus.SUCCEEDED,
                ExecutionStatus.FAILED,
                ExecutionStatus.CANCELLED,
            ]
        ),
    )
)
Index(
    "executions_due_idx",
    ExecutionRow.next_attempt_at,
    ExecutionRow.created_at,
    postgresql_where=ExecutionRow.work_kind.is_not(None),
)
Index(
    "executions_active_processing_idx",
    ExecutionRow.model_id,
    ExecutionRow.operation,
    ExecutionRow.lease_expires_at,
    postgresql_where=and_(
        ExecutionRow.work_kind == WorkKind.PROCESSING,
        ExecutionRow.fencing_token.is_not(None),
    ),
)
for name, column in (
    ("executions_deadline_idx", ExecutionRow.deadline_at),
    ("executions_report_deadline_idx", ExecutionRow.terminal_report_deadline_at),
):
    Index(name, column, postgresql_where=ExecutionRow.work_kind.is_not(None))
Index(
    "executions_lease_idx",
    ExecutionRow.lease_expires_at,
    postgresql_where=ExecutionRow.fencing_token.is_not(None),
)
Index(
    "executions_finished_idx",
    ExecutionRow.updated_at,
    postgresql_where=ExecutionRow.work_kind.is_(None),
)
