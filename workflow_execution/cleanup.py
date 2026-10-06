from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


# Keep cancellation-resistant cleanup alive until it actually finishes.
_pending_cleanup: set[asyncio.Future[None]] = set()


async def close_resource(
    close: Callable[[], Awaitable[None]],
    *,
    timeout_seconds: float = 10,
) -> None:
    task = asyncio.ensure_future(close())
    waiter = asyncio.create_task(asyncio.wait({task}, timeout=timeout_seconds))
    cancellation: asyncio.CancelledError | None = None
    while not waiter.done():
        try:
            await asyncio.shield(waiter)
        except asyncio.CancelledError as error:
            cancellation = error
    if not task.done():
        task.cancel()
        _pending_cleanup.add(task)
        task.add_done_callback(_release_cleanup)
        if cancellation is not None:
            raise cancellation
        raise TimeoutError("workflow resource cleanup timed out")
    if cancellation is not None:
        _consume_exception(task)
        raise cancellation
    if task.cancelled():
        raise RuntimeError("workflow resource cleanup was cancelled")
    task.result()


def _consume_exception(task: asyncio.Future[None]) -> None:
    if not task.cancelled():
        task.exception()


def _release_cleanup(task: asyncio.Future[None]) -> None:
    _pending_cleanup.discard(task)
    _consume_exception(task)
