"""Claiming due work and renewing its lease, under per-model advisory locks."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import and_, func, or_, select

from model_service.scheduler.internal_state import (
    ExecutionRecord,
    RenewDirective,
    WorkClaim,
    WorkKind,
)
from model_service.scheduler.store.rows import ExecutionRow


if TYPE_CHECKING:
    from datetime import datetime, timedelta

    from sqlalchemy.ext.asyncio import AsyncSession

    from model_service.scheduler.store.postgres.base import PostgresDatabase


logger = logging.getLogger(__name__)


class ClaimOperations:
    """Claiming due work and renewing its lease; MaintenanceOperations recovers expired leases."""

    def __init__(self, db: PostgresDatabase) -> None:
        """Keep the database these operations run on.

        Args:
            db: Connection pool, transactions and row mapping.
        """
        self._db = db

    async def claim_due(
        self,
        owner: str,
        lease_duration: timedelta,
    ) -> WorkClaim | None:
        """Lease due provider work before capacity-limited processing using database time.

        Args:
            owner: ID of the claiming worker.
            lease_duration: How long the claim stays valid without renewal.

        Returns:
            The claim, or None when nothing is due or the selected row's context is
            unreadable.

        Raises:
            ExecutionRepositoryUnavailableError: The database is unavailable.
        """
        async with self._db.transaction() as session:
            now = await self._db.database_now(session)
            row = await self._select_finish(session, now)
            if row is None:
                row = await self._select_processing(session, now)
            if row is None:
                return None
            execution = await self._claimable(session, row, now)
            if execution is None:
                return None
            claimed = self._db.transitions.leases.claim(
                execution,
                owner=owner,
                fencing_token=str(uuid4()),
                lease_duration=lease_duration,
                now=now,
            )
            await self._db.persist_state(session, row, claimed.execution)
        return claimed.claim

    async def renew(
        self,
        execution_id: str,
        fencing_token: str,
        lease_duration: timedelta,
    ) -> RenewDirective:
        """Lock a record, check its token and deadline, and commit a valid lease extension or stop decision.

        Args:
            execution_id: ID of the claimed execution.
            fencing_token: Token of the renewing attempt.
            lease_duration: Extension to apply from now.

        Returns:
            The directive the worker must follow.

        Raises:
            UnknownExecutionError: No execution has this ID.
            ExecutionRepositoryUnavailableError: The database is unavailable.
        """
        async with self._db.transaction() as session:
            row = await self._db.fetch(session, execution_id, for_update=True)
            decision = self._db.transitions.leases.decide_renewal(
                self._db.record(row),
                fencing_token=fencing_token,
                lease_duration=lease_duration,
                now=await self._db.database_now(session),
            )
            if decision.execution is not None:
                await self._db.persist_state(session, row, decision.execution)
            return decision.directive

    async def _select_finish(
        self, session: AsyncSession, now: datetime
    ) -> ExecutionRow | None:
        """Lock the earliest eligible provider attempt with SKIP LOCKED, excluding expired work deadlines.

        Args:
            session: Session of the caller's transaction.
            now: Current database time.

        Returns:
            The highest-priority due provider row, or None.
        """
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
            .order_by(
                ExecutionRow.priority.desc(),
                ExecutionRow.next_attempt_at,
                ExecutionRow.created_at,
            )
            .with_for_update(skip_locked=True)
            .limit(1)
        )

    async def _select_processing(
        self, session: AsyncSession, now: datetime
    ) -> ExecutionRow | None:
        """Lock the oldest eligible processing row within its model's capacity.

        A transaction-scoped advisory lock per model serializes the capacity check
        across scheduler replicas; a model whose lock is held elsewhere is skipped.

        Args:
            session: Session of the caller's transaction.
            now: Current database time.

        Returns:
            The locked row, by model priority and age, or None.
        """
        eligible = (
            ExecutionRow.work_kind == WorkKind.PROCESSING,
            ExecutionRow.fencing_token.is_(None),
            ExecutionRow.next_attempt_at <= now,
            ExecutionRow.deadline_at > now,
        )
        models = await session.scalars(
            select(ExecutionRow.model_id)
            .where(*eligible)
            .group_by(ExecutionRow.model_id)
            .order_by(
                func.max(ExecutionRow.priority).desc(),
                func.min(ExecutionRow.created_at),
            )
        )
        for model_id in models.all():
            locked = await session.scalar(
                select(
                    func.pg_try_advisory_xact_lock(
                        func.hashtextextended(f"builder:{model_id}", 0)
                    )
                )
            )
            if not locked:
                continue
            active = (
                await session.execute(
                    select(func.count())
                    .select_from(ExecutionRow)
                    .where(
                        ExecutionRow.model_id == model_id,
                        ExecutionRow.work_kind == WorkKind.PROCESSING,
                        ExecutionRow.fencing_token.is_not(None),
                        ExecutionRow.lease_expires_at > now,
                    )
                )
            ).scalar_one()
            if active >= self._db.per_model_limit:
                continue
            row = await session.scalar(
                select(ExecutionRow)
                .where(ExecutionRow.model_id == model_id, *eligible)
                .order_by(
                    ExecutionRow.priority.desc(),
                    ExecutionRow.next_attempt_at,
                    ExecutionRow.created_at,
                )
                .with_for_update(skip_locked=True)
                .limit(1)
            )
            if row is not None:
                return row
        return None

    async def _claimable(
        self, session: AsyncSession, row: ExecutionRow, now: datetime
    ) -> ExecutionRecord | None:
        """Decode a row with its provider context for claiming.

        A record that cannot be read fails, not the queue: it is ended in place, and
        its stored bytes stay in case the key was wrong.

        Args:
            session: Session of the caller's transaction.
            row: Locked candidate row.
            now: Current database time.

        Returns:
            The record with its decrypted context, or None if the context is unreadable.
        """
        try:
            return self._db.record(row, protected=True)
        except ValueError:  # Includes ProtectedContextError: a bad envelope or key.
            logger.error("execution %s has an unreadable provider context", row.id)
            # Not _persist_state: the stored bytes stay, in case the key was wrong.
            self._db.apply_state(
                row, self._db.transitions.outcomes.unreadable(self._db.record(row), now)
            )
            await session.flush()
            return None
