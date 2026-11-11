"""Settling an attempt's outcome and recording provider-finish mutations."""

from __future__ import annotations

from model_service.scheduler.internal_state import (
    ExecutionRecord,
    Settlement,
)
from model_service.scheduler.store.postgres.base import (
    PostgresBase,
)


class SettlementOperations(PostgresBase):
    """Settling attempts and provider-finish mutations."""

    async def settle(
        self,
        execution_id: str,
        fencing_token: str,
        settlement: Settlement,
    ) -> ExecutionRecord:
        """Lock and validate an attempt's outcome, committing deadline changes before raising a stale-claim rejection.

        Args:
            execution_id: ID of the claimed execution.
            fencing_token: Token of the settling attempt.
            settlement: Outcome of the attempt.

        Returns:
            The record after the outcome.

        Raises:
            UnknownExecutionError: No execution has this ID.
            StaleWorkClaimError: The token or lease is no longer current, the deadline passed, or the outcome does not fit the work.
            ExecutionRepositoryUnavailableError: The database is unavailable.
        """
        async with self._transaction() as session:
            row = await self._fetch(session, execution_id, for_update=True)
            decision = self._transitions.decide_settlement(
                self._record(row),
                fencing_token,
                settlement,
                await self._database_now(session),
            )
            if decision.execution is not None:
                await self._persist_state(session, row, decision.execution)
        return decision.accepted_execution()

    async def mark_finish_mutation_started(
        self,
        execution_id: str,
        fencing_token: str,
    ) -> None:
        """Lock and validate the provider claim before committing its external-mutation marker.

        Args:
            execution_id: ID of the claimed execution.
            fencing_token: Token of the provider attempt.

        Raises:
            UnknownExecutionError: No execution has this ID.
            StaleWorkClaimError: The claim is not provider work or is no longer current.
            ExecutionRepositoryUnavailableError: The database is unavailable.
        """
        async with self._transaction() as session:
            row = await self._fetch(session, execution_id, for_update=True)
            marked = self._transitions.mark_finish_mutation_started(
                self._record(row),
                fencing_token,
                await self._database_now(session),
            )
            await self._persist_state(session, row, marked)
