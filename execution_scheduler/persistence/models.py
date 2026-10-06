from __future__ import annotations

from datetime import datetime  # noqa: TC003 - SQLAlchemy resolves mapped annotations.
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
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

from execution_scheduler.contracts import ExecutionStatus
from execution_scheduler.domain import WorkKind
from workflow_execution.contracts import (
    WorkflowInvocation,
    WorkflowResult,
)
from workflow_execution.serialization import (
    invocation_from_dict,
    invocation_to_dict,
    result_from_dict,
    result_to_dict,
)


if TYPE_CHECKING:
    from sqlalchemy.engine import Dialect


DEFAULT_SCHEMA = "model_service"


class WorkflowInvocationType(TypeDecorator[WorkflowInvocation]):
    impl = JSONB(none_as_null=True)
    cache_ok = True

    def process_bind_param(
        self, value: WorkflowInvocation | None, dialect: Dialect
    ) -> dict[str, object] | None:
        return invocation_to_dict(value) if value is not None else None

    def process_result_value(
        self, value: object, dialect: Dialect
    ) -> WorkflowInvocation | None:
        return invocation_from_dict(value) if value is not None else None


class WorkflowResultType(TypeDecorator[WorkflowResult]):
    impl = JSONB(none_as_null=True)
    cache_ok = True

    def process_bind_param(
        self, value: WorkflowResult | None, dialect: Dialect
    ) -> dict[str, object] | None:
        return result_to_dict(value) if value is not None else None

    def process_result_value(
        self, value: object, dialect: Dialect
    ) -> WorkflowResult | None:
        return result_from_dict(value) if value is not None else None


class Base(DeclarativeBase):
    pass


class ExecutionRow(Base):
    __tablename__ = "executions"
    __table_args__ = (
        UniqueConstraint("provider", "submission_key"),
        {"schema": DEFAULT_SCHEMA},
    )

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True)
    provider: Mapped[str] = mapped_column(Text)
    submission_key: Mapped[str] = mapped_column(Text)
    request_fingerprint: Mapped[str] = mapped_column(Text)
    workflow_id: Mapped[str] = mapped_column(Text)
    status: Mapped[ExecutionStatus] = mapped_column(
        Enum(
            ExecutionStatus,
            native_enum=False,
            create_constraint=True,
            length=None,
            values_callable=lambda enum: [status.value for status in enum],
        )
    )
    invocation: Mapped[WorkflowInvocation] = mapped_column(WorkflowInvocationType())
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
    result: Mapped[WorkflowResult | None] = mapped_column(WorkflowResultType())
    error_code: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.clock_timestamp()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.clock_timestamp()
    )


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
    ExecutionRow.workflow_id,
    ExecutionRow.lease_expires_at,
    postgresql_where=and_(
        ExecutionRow.work_kind == WorkKind.PROCESSING,
        ExecutionRow.fencing_token.is_not(None),
    ),
)
