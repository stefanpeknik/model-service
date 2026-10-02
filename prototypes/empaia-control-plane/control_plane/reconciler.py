from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from control_plane.domain import AttemptOwnershipError


if TYPE_CHECKING:
    from control_plane.store import SQLiteExecutionStore


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReconcileResult:
    timed_out: int
    recovered_attempts: int
    cancelled: int


class ExecutionReconciler:
    """Periodically fences stale workers and applies durable timeout policy."""

    def __init__(
        self,
        store: SQLiteExecutionStore,
        max_initial_capacity_wait_seconds: int | None,
        worker_stale_after_seconds: float,
        retry_backoff_seconds: float,
    ) -> None:
        if worker_stale_after_seconds <= 0 or retry_backoff_seconds < 0:
            raise ValueError("reconciliation durations are invalid")
        self._store = store
        self._max_initial_capacity_wait_seconds = max_initial_capacity_wait_seconds
        self._worker_stale_after_seconds = worker_stale_after_seconds
        self._retry_backoff_seconds = retry_backoff_seconds

    def run_once(self) -> ReconcileResult:
        now = datetime.now(UTC)
        cancelled = self._store.settle_requested_cancellations()
        timed_out = self._store.expire_timeouts(self._max_initial_capacity_wait_seconds)
        stale_before = now - timedelta(seconds=self._worker_stale_after_seconds)
        stale = self._store.list_stale_attempts(stale_before)
        recovered = 0
        for execution in stale:
            if execution.current_attempt_id is None:
                continue
            try:
                self._store.recover_attempt(
                    execution.id,
                    execution.current_attempt_id,
                    stale_before,
                    now + timedelta(seconds=self._retry_backoff_seconds),
                )
            except AttemptOwnershipError:
                # A heartbeat, completion, or another reconciler won after the scan.
                continue
            recovered += 1
        return ReconcileResult(
            timed_out=len(timed_out),
            recovered_attempts=recovered,
            cancelled=len(cancelled),
        )

    async def run(self, stop: asyncio.Event, interval_seconds: float) -> None:
        if interval_seconds <= 0:
            raise ValueError("reconciliation interval must be positive")
        while not stop.is_set():
            try:
                await asyncio.to_thread(self.run_once)
            except Exception:
                logger.exception("unexpected execution reconciliation failure")
            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
