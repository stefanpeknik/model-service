"""Check the shared repository behavior against the in-memory adapter."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from execution_scheduler.domain import RetryPolicy
from execution_scheduler.persistence.memory import InMemoryExecutionRepository
from tests.repository_contract import ExecutionRepositoryContract


class InMemoryRepositoryContractTests(
    ExecutionRepositoryContract, unittest.IsolatedAsyncioTestCase
):
    """Exercise the shared repository behavior against the in-memory adapter."""

    def setUp(self) -> None:
        """Create isolated fixtures and clients for the shared repository behavior against the in-memory adapter.

        Returns:
            None: No return value.
        """
        self.now = datetime.now(UTC)
        policy = RetryPolicy(3, timedelta(0), timedelta(0))
        self.repository = InMemoryExecutionRepository(
            clock=lambda: self.now, processing_policy=policy, finish_policy=policy
        )

    async def _advance_time(self, duration: timedelta) -> None:
        """Advance the repository test clock or wait for database time according to the concrete adapter.

        Args:
            duration: Amount of time to advance or wait in a deterministic lifecycle
                scenario.

        Returns:
            None: No return value.
        """
        self.now += duration
