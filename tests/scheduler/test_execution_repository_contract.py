"""Check the shared repository behavior against the in-memory adapter."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from model_service.scheduler.domain import RetryPolicy
from tests.support.memory_repository import InMemoryExecutionRepository
from tests.support.repository_contract import (
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
        self.payload_root = self.repository.payload_root

    async def _advance_time(self, duration: timedelta) -> None:
        self.now += duration
