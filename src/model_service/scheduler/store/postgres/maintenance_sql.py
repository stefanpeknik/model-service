"""Expiring deadlines and recovering abandoned claims."""

from __future__ import annotations

from typing import TYPE_CHECKING

from sqlalchemy import and_, func, or_, select

from model_service.scheduler.internal_state import WorkKind
from model_service.scheduler.status import ExecutionStatus
from model_service.scheduler.store.rows import ExecutionRow


if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession

    from model_service.scheduler.store.postgres.base import PostgresDatabase


class MaintenanceOperations:
    """Expiring deadlines and recovering abandoned claims."""

    def __init__(self, db: PostgresDatabase) -> None:
        """Keep the database these operations run on.

        Args:
            db: Connection pool, transactions and row mapping.
        """
        self._db = db

    async def maintain(self) -> None:
        """Expire deadlines and recover abandoned claims.

        Raises:
            ExecutionRepositoryUnavailableError: The database is unavailable.
        """
        async with self._db.transaction() as session:
            now = await self._db.database_now(session)
            await self._expire_deadlines(session, now)
            await self._recover_expired_leases(session, now)

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
            recovered = self._db.transitions.expiry.recover_expired(
                self._db.record(row), now
            )
            await self._db.persist_state(session, row, recovered)

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
            await self._db.persist_state(
                session,
                row,
                self._db.transitions.expiry.deadline(self._db.record(row), now),
            )
