"""Per-step timing of a run, to tell slide reads from model waits in the logs."""

from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Callable


class Timings:
    """Call counts and total seconds per stage (source, each step, each sink).

    Safe to update from the read pool's threads.
    """

    def __init__(self) -> None:
        """Start with no stage timed."""
        self._calls: dict[str, list[float]] = defaultdict(lambda: [0, 0.0])
        self._lock = Lock()

    def add(self, stage: str, seconds: float) -> None:
        """Record one call of a stage.

        Args:
            stage: The stage's name, e.g. `slide_tiles` or `model m1`.
            seconds: How long the call took.
        """
        with self._lock:
            entry = self._calls[stage]
            entry[0] += 1
            entry[1] += seconds

    def time[T](self, stage: str, function: Callable[..., T], *args: Any) -> T:
        """Call a function and record how long it took.

        Args:
            stage: The stage's name.
            function: What to call.
            *args: Its arguments.

        Returns:
            What it returned.
        """
        began = time.perf_counter()
        try:
            return function(*args)
        finally:
            self.add(stage, time.perf_counter() - began)

    def __str__(self) -> str:
        """Mean milliseconds per call of each stage, in the order first timed."""
        with self._lock:
            parts = [
                f"{stage} {seconds / count * 1000:.1f} ms"
                for stage, (count, seconds) in self._calls.items()
                if count
            ]
        return ", ".join(parts) or "none"
