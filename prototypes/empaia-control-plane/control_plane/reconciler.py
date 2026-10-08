"""Apply prototype cancellation, timeout, and stale-attempt recovery policies periodically.

Historical SQLite control-plane prototype. The current production scheduler lives in
execution_scheduler/ and uses PostgreSQL; these lifecycle rules are not its
specification.
"""

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
    """Count timeouts, recovered worker attempts, and cancellations in one pass.

    Attributes:
        timed_out: Count of prototype executions whose deadlines were applied.
        recovered_attempts: Count of stale prototype attempts fenced/retried.
        cancelled: Count of prototype cancellations settled in this pass.
    """

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
        """Validate heartbeat/retry timing and retain the storage and capacity-wait policy.

        Args:
            store: Historical prototype transactional SQLite execution store.
            max_initial_capacity_wait_seconds: Optional prototype wait limit before the
                first compute claim.
            worker_stale_after_seconds: Prototype heartbeat age after which an attempt
                may be recovered.
            retry_backoff_seconds: Delay before a recovered prototype attempt becomes
                dispatchable.

        Raises:
            ValueError: A value fails the validation described above.
        """
        if worker_stale_after_seconds <= 0 or retry_backoff_seconds < 0:
            raise ValueError("reconciliation durations are invalid")
        self._store = store
        self._max_initial_capacity_wait_seconds = max_initial_capacity_wait_seconds
        self._worker_stale_after_seconds = worker_stale_after_seconds
        self._retry_backoff_seconds = retry_backoff_seconds

    def run_once(self) -> ReconcileResult:
        """Settle cancellations, expire timeouts, and recover stale attempts while tolerating concurrent worker ownership changes.

        Returns:
            ReconcileResult: Counts of timeouts, recovered attempts, and cancellations
                in this pass.
        """
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
        """Run synchronous reconciliation in a thread at a positive interval until shutdown.

        Args:
            stop: Event signalled when the owning worker/resolver lifetime should end.
            interval_seconds: Positive polling interval between resolver or reconciler
                passes.

        Returns:
            None: No return value.

        Raises:
            ValueError: A value fails the validation described above.
        """
        if interval_seconds <= 0:
            raise ValueError("reconciliation interval must be positive")
        while not stop.is_set():
            try:
                await asyncio.to_thread(self.run_once)
            except Exception:
                logger.exception("unexpected execution reconciliation failure")
            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=interval_seconds)
