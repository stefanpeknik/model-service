"""Check the shared repository behavior against the in-memory adapter."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from execution_scheduler.domain import RetryPolicy
from tests.memory_repository import InMemoryExecutionRepository
from tests.repository_contract import (
    INFERENCE_LIMIT,
    INFERENCE_RETENTION,
    ExecutionRepositoryContract,
)


class InMemoryRepositoryContractTests(
    ExecutionRepositoryContract, unittest.IsolatedAsyncioTestCase
):
    def setUp(self) -> None:
        self.now = datetime.now(UTC)
        policy = RetryPolicy(3, timedelta(0), timedelta(0))
        self.repository = InMemoryExecutionRepository(
            clock=lambda: self.now,
            per_model_inference_limit=INFERENCE_LIMIT,
            inference_retention=INFERENCE_RETENTION,
            processing_policy=policy,
            finish_policy=policy,
        )

    async def _advance_time(self, duration: timedelta) -> None:
        self.now += duration
