"""Pass a builder's progress to its provider: throttled, and never failing the run.

A builder reports as often as it likes; only the latest value is kept, and it
is sent once per interval when it moved by at least a percentage point. A
failed send is logged once and the run goes on. Sending stops when the run
ends, before the provider's access is closed.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable


logger = logging.getLogger(__name__)

# Smallest change worth a send: one percentage point.
MIN_STEP = 0.01


class _ProgressReporter:
    """Keeps a run's latest progress and sends it on a fixed interval."""

    def __init__(
        self, send: Callable[[float], Awaitable[None]], *, interval_seconds: float
    ) -> None:
        """Keep where to send progress and how often.

        Args:
            send: Hands one fraction to the provider.
            interval_seconds: Time between two sends.
        """
        self._send = send
        self._interval_seconds = interval_seconds
        self._latest: float | None = None
        self._sent: float | None = None
        self._failed = False

    def report(self, fraction: float) -> None:
        """Record the latest progress; it is sent on the next interval.

        Args:
            fraction: Share of the work done, from 0 to 1.
        """
        self._latest = fraction

    async def run(self) -> None:
        """Send the latest progress every interval while it moved; runs until cancelled."""
        while True:
            await asyncio.sleep(self._interval_seconds)
            latest = self._latest
            if latest is None or (
                self._sent is not None and abs(latest - self._sent) < MIN_STEP
            ):
                continue
            try:
                await self._send(latest)
            except Exception as error:
                if not self._failed:
                    self._failed = True
                    logger.warning(
                        "progress could not be sent: %s", type(error).__name__
                    )
                continue
            self._sent = latest


@asynccontextmanager
async def reporting_progress(
    send: Callable[[float], Awaitable[None]], *, interval_seconds: float
) -> AsyncIterator[Callable[[float], None]]:
    """Send progress in the background while the block runs.

    Args:
        send: Hands one fraction to the provider.
        interval_seconds: Time between two sends.

    Yields:
        The function the builder reports progress with; after the block it
        still accepts reports but sends none.
    """
    reporter = _ProgressReporter(send, interval_seconds=interval_seconds)
    task = asyncio.create_task(reporter.run())
    try:
        yield reporter.report
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
