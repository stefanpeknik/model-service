"""Periodic maintenance: expired leases, passed deadlines, old payloads."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from sqlalchemy import and_, delete, func, or_, select

from model_service.scheduler.records import (
    WorkKind,
)
from model_service.scheduler.status import ExecutionStatus, Operation
from model_service.scheduler.store.postgres.base import (
    _ORPHAN_SWEEP_SECONDS,
    ORPHAN_AGE,
    PostgresBase,
    _delete_after_commit,
    logger,
)
from model_service.scheduler.store.rows import ExecutionRow


if TYPE_CHECKING:
    from datetime import datetime

    from sqlalchemy.ext.asyncio import AsyncSession


class MaintenanceOperations(PostgresBase):
    """Lease recovery, deadline expiry and payload sweeps."""

    async def maintain(self) -> None:
        """Expire deadlines, recover abandoned claims and delete expired inference results.

        Every few minutes it also removes payload files no execution owns; a failed
        sweep is logged, not raised.

        Raises:
            ExecutionRepositoryUnavailableError: The database is unavailable.
        """
        async with self._transaction() as session:
            now = await self._database_now(session)
            await self._expire_deadlines(session, now)
            await self._recover_expired_leases(session, now)
            await self._delete_expired_inference(session, now)
        if time.monotonic() >= self._next_orphan_sweep:
            self._next_orphan_sweep = time.monotonic() + _ORPHAN_SWEEP_SECONDS
            try:
                await self._payloads.sweep_orphans(
                    self._existing_ids, ORPHAN_AGE.total_seconds()
                )
            except OSError as error:
                logger.warning("payload sweep failed: %s", type(error).__name__)

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
            recovered = self._transitions.recover_expired(self._record(row), now)
            await self._persist_state(session, row, recovered)

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
            await self._persist_state(
                session, row, self._transitions.deadline(self._record(row), now)
            )

    async def _delete_expired_inference(
        self, session: AsyncSession, now: datetime
    ) -> None:
        """Delete up to 1000 finished inference executions past retention.

        Their files are removed once the deletion commits.

        Args:
            session: Session of the caller's transaction.
            now: Current database time.
        """
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
        deleted = await session.execute(
            delete(ExecutionRow)
            .where(ExecutionRow.id.in_(expired))
            .returning(ExecutionRow.request_path, ExecutionRow.response_path)
        )
        _delete_after_commit(
            session, [path for row in deleted for path in row if path is not None]
        )
