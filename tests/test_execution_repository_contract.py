from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from execution_scheduler.domain import RetryPolicy
from execution_scheduler.repository import InMemoryExecutionRepository
from tests.repository_contract import ExecutionRepositoryContract


class InMemoryRepositoryContractTests(
    ExecutionRepositoryContract, unittest.IsolatedAsyncioTestCase
):
    def setUp(self) -> None:
        self.now = datetime.now(UTC)
        policy = RetryPolicy(3, timedelta(0), timedelta(0))
        self.repository = InMemoryExecutionRepository(
            clock=lambda: self.now, processing_policy=policy, finish_policy=policy
        )

    async def _advance_time(self, duration: timedelta) -> None:
        self.now += duration
