"""Request batching on the replica's event loop, the way `@serve.batch` does it.

Ray's `serve.batch` runs only inside a Serve replica; this one runs on any
event loop, so the host core works off Serve with the same behaviour: the
first waiting request starts a `wait_s` timer, and a batch leaves when it is
full or the timer ends, never larger than `max_size`.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable


class Batcher[T, R]:
    """Groups concurrent `submit` calls into batches for one function."""

    def __init__(
        self, run: Callable[[list[T]], list[R]], *, max_size: int, wait_s: float
    ) -> None:
        """Bind the batch function and the batch limits.

        Args:
            run: Turns a batch of items into one result per item, in order.
                It runs on the event loop, like a `@serve.batch` method.
            max_size: The largest batch.
            wait_s: How long the first waiting item waits for others.
        """
        self._run = run
        self._max_size = max_size
        self._wait_s = wait_s
        self._pending: list[tuple[T, asyncio.Future[R]]] = []
        self._full = asyncio.Event()
        self._drainer: asyncio.Task[None] | None = None

    async def submit(self, item: T) -> R:
        """Add one item to the next batch and wait for its result.

        Args:
            item: One item.

        Returns:
            The item's result.

        Raises:
            Exception: Whatever the batch function raised for its batch.
        """
        future: asyncio.Future[R] = asyncio.get_running_loop().create_future()
        self._pending.append((item, future))
        if self._drainer is None or self._drainer.done():
            self._full = asyncio.Event()
            self._drainer = asyncio.create_task(self._drain())
        if len(self._pending) >= self._max_size:
            self._full.set()
        return await future

    async def _drain(self) -> None:
        """Run batches until nothing waits."""
        while self._pending:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._full.wait(), self._wait_s)
            batch = self._pending[: self._max_size]
            del self._pending[: self._max_size]
            if len(self._pending) < self._max_size:
                self._full.clear()
            self._settle(batch)

    def _settle(self, batch: list[tuple[T, asyncio.Future[R]]]) -> None:
        """Run one batch and hand each caller its result or the batch's error."""
        try:
            results = self._run([item for item, _ in batch])
            if len(results) != len(batch):
                raise RuntimeError(
                    f"the batch function returned {len(results)} results "
                    f"for {len(batch)} items"
                )
        except Exception as error:
            for _, future in batch:
                if not future.done():
                    future.set_exception(error)
            return
        for (_, future), result in zip(batch, results, strict=True):
            if not future.done():
                future.set_result(result)
