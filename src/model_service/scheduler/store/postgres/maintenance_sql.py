"""Periodic maintenance: expired leases and passed deadlines."""

from __future__ import annotations

from model_service.scheduler.store.postgres.base import PostgresBase


class MaintenanceOperations(PostgresBase):
    """Lease recovery and deadline expiry."""

    async def maintain(self) -> None:
        """Expire deadlines and recover abandoned claims.

        Raises:
            ExecutionRepositoryUnavailableError: The database is unavailable.
        """
        async with self._transaction() as session:
            now = await self._database_now(session)
            await self._expire_deadlines(session, now)
            await self._recover_expired_leases(session, now)
