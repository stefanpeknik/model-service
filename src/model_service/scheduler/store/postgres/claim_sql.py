"""Claiming due work and renewing its lease, under per-model advisory locks."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import and_, func, or_, select

from model_service.scheduler.execution_records import (
    ExecutionRecord,
    RenewDirective,
    WorkClaim,
    WorkKind,
)
from model_service.scheduler.executions import (
    ResultUnavailableError,
)
from model_service.scheduler.status import ExecutionStatus, Operation
from model_service.scheduler.store.payloads import PayloadMissingError, StoredPayload
from model_service.scheduler.store.rows import ExecutionRow


if TYPE_CHECKING:
    from collections.abc import Collection
    from datetime import datetime, timedelta

    from sqlalchemy.ext.asyncio import AsyncSession


from model_service.scheduler.store.postgres.base import (
    _stored_request,
    logger,
)
from model_service.scheduler.store.postgres.maintenance_sql import MaintenanceOperations


class ClaimOperations(MaintenanceOperations):
    """Claims recover expired leases and passed deadlines first, as maintenance does."""

    async def claim_due(
        self,
        owner: str,
        lease_duration: timedelta,
        operations: Collection[Operation] | None = None,
    ) -> WorkClaim | None:
        """Run maintenance, then lease due provider work before capacity-limited processing using database time.

        Args:
            owner: ID of the claiming worker.
            lease_duration: How long the claim stays valid without renewal.
            operations: Operations to claim from, or None for all. Provider work is claimed only when builders are allowed.

        Returns:
            The claim, with its payload for inference, or None when nothing is due or
            the selected row's context is unreadable.

        Raises:
            ExecutionRepositoryUnavailableError: The database is unavailable.
        """
        allowed = frozenset(Operation if operations is None else operations)
        async with self._transaction() as session:
            now = await self._database_now(session)
            await self._expire_deadlines(session, now)
            await self._recover_expired_leases(session, now)
            row = None
            if Operation.BUILDER in allowed:
                row = await self._select_finish(session, now)
            if row is None:
                rows = await self._select_processing(session, now, allowed, limit=1)
                row = rows[0] if rows else None
            if row is None:
                return None
            execution = await self._claimable(session, row, now)
            if execution is None:
                return None
            claimed = self._transitions.claim(
                execution,
                owner=owner,
                fencing_token=str(uuid4()),
                lease_duration=lease_duration,
                now=now,
            )
            await self._persist_state(session, row, claimed.execution)
            request = _stored_request(row)
        if claimed.claim.operation is not Operation.INFERENCE:
            return claimed.claim
        return await self._with_request(claimed.claim, request)

    async def claim_inference(
        self, owner: str, lease_duration: timedelta, limit: int
    ) -> list[WorkClaim]:
        """Lease up to `limit` due inference attempts, with payloads, in one transaction.

        Payload files are read after commit, not while the row locks are held.

        Args:
            owner: ID of the claiming worker.
            lease_duration: How long each claim stays valid without renewal.
            limit: Maximum number of claims.

        Returns:
            The claims; empty when nothing is due or limit is not positive.

        Raises:
            ExecutionRepositoryUnavailableError: The database is unavailable.
        """
        if limit <= 0:
            return []
        async with self._transaction() as session:
            now = await self._database_now(session)
            await self._expire_deadlines(session, now)
            await self._recover_expired_leases(session, now)
            rows = await self._select_processing(
                session, now, frozenset({Operation.INFERENCE}), limit=limit
            )
            claims = []
            for row in rows:
                execution = await self._claimable(session, row, now)
                if execution is None:
                    continue
                claimed = self._transitions.claim(
                    execution,
                    owner=owner,
                    fencing_token=str(uuid4()),
                    lease_duration=lease_duration,
                    now=now,
                )
                self._apply_state(row, claimed.execution)
                claims.append((claimed.claim, _stored_request(row)))
            if not claims:
                return []
            await session.flush()
        # Read the files after commit, not while holding the rows' locks.
        return list(
            await asyncio.gather(
                *(self._with_request(claim, request) for claim, request in claims)
            )
        )

    async def inference_content(self, execution_id: str) -> bytes:
        """Return the stored response of a succeeded inference execution.

        Args:
            execution_id: ID of the inference execution.

        Returns:
            The response body.

        Raises:
            UnknownExecutionError: No execution has this ID.
            ResultUnavailableError: The execution has no stored response or its file cannot be read.
            ExecutionRepositoryUnavailableError: The database is unavailable.
        """
        async with self._transaction() as session:
            row = await self._fetch(session, execution_id)
            response = None
            if (
                row.operation is Operation.INFERENCE
                and row.status is ExecutionStatus.SUCCEEDED
                and row.response_path is not None
                and row.response_size is not None
            ):
                response = StoredPayload(row.response_path, row.response_size)
        if response is None:
            raise ResultUnavailableError("execution has no inference response")
        try:
            return await self._payloads.read(response)
        except (PayloadMissingError, OSError) as error:
            raise ResultUnavailableError("inference response is unavailable") from error

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
        async with self._transaction() as session:
            row = await self._fetch(session, execution_id, for_update=True)
            decision = self._transitions.decide_renewal(
                self._record(row),
                fencing_token=fencing_token,
                lease_duration=lease_duration,
                now=await self._database_now(session),
            )
            if decision.execution is not None:
                await self._persist_state(session, row, decision.execution)
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
        self,
        session: AsyncSession,
        now: datetime,
        operations: frozenset[Operation],
        *,
        limit: int,
    ) -> list[ExecutionRow]:
        """Lock up to `limit` oldest eligible processing rows within each model's capacity.

        Builders and raw inference have separate limits per model. A
        transaction-scoped advisory lock per (operation, model) serializes the
        capacity check across scheduler replicas; a model whose lock is held elsewhere is skipped.

        Args:
            session: Session of the caller's transaction.
            now: Current database time.
            operations: Operations to select from.
            limit: Maximum number of rows.

        Returns:
            The locked rows, by model priority and age.
        """
        eligible = (
            ExecutionRow.work_kind == WorkKind.PROCESSING,
            ExecutionRow.fencing_token.is_(None),
            ExecutionRow.next_attempt_at <= now,
            ExecutionRow.deadline_at > now,
            ExecutionRow.operation.in_(operations),
        )
        targets = await session.execute(
            select(ExecutionRow.operation, ExecutionRow.model_id)
            .where(*eligible)
            .group_by(ExecutionRow.operation, ExecutionRow.model_id)
            .order_by(
                func.max(ExecutionRow.priority).desc(),
                func.min(ExecutionRow.created_at),
            )
        )
        selected: list[ExecutionRow] = []
        for operation, model_id in targets.tuples():
            locked = await session.scalar(
                select(
                    func.pg_try_advisory_xact_lock(
                        func.hashtextextended(f"{operation.value}:{model_id}", 0)
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
                        ExecutionRow.operation == operation,
                        ExecutionRow.work_kind == WorkKind.PROCESSING,
                        ExecutionRow.fencing_token.is_not(None),
                        ExecutionRow.lease_expires_at > now,
                    )
                )
            ).scalar_one()
            wanted = min(self._limits[operation] - active, limit - len(selected))
            if wanted <= 0:
                continue
            candidates = await session.scalars(
                select(ExecutionRow)
                .where(
                    ExecutionRow.model_id == model_id,
                    ExecutionRow.operation == operation,
                    *eligible,
                )
                .order_by(
                    ExecutionRow.priority.desc(),
                    ExecutionRow.next_attempt_at,
                    ExecutionRow.created_at,
                )
                .with_for_update(skip_locked=True)
                .limit(wanted)
            )
            selected.extend(candidates)
            if len(selected) >= limit:
                break
        return selected

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
            return self._record(row, protected=True)
        except ValueError:  # Includes ProtectedContextError: a bad envelope or key.
            logger.error("execution %s has an unreadable provider context", row.id)
            # Not _persist_state: the stored bytes stay, in case the key was wrong.
            self._apply_state(row, self._transitions.unreadable(self._record(row), now))
            await session.flush()
            return None
