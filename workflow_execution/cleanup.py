"""Close workflow resources with bounded waiting while preserving caller cancellation.

Cancellation-resistant cleanup tasks remain strongly referenced until they finish, and
their exceptions are consumed to avoid unhandled-task warnings.
"""

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
    """Start cleanup and shield its bounded wait from caller cancellation.

    After waiting, propagate cancellation or cleanup failure. On timeout, cancel cleanup
    and retain the task until completion, since the resource may resist cancellation.

    Args:
        close: Async callback that releases the acquired resource.
        timeout_seconds: Maximum wait for cleanup, polling, or a classified operation.

    Returns:
        None: No return value.

    Raises:
        RuntimeError: Cleanup ends by cancelling its own task.
        TimeoutError: Cleanup exceeds timeout_seconds without a pending caller
            cancellation.
        asyncio.CancelledError: The caller is cancelled; bounded cleanup finishes or
            times out before cancellation propagates.
        Exception: The close callback fails; its original exception propagates when no
            caller cancellation takes precedence.
    """
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
    """Retrieve a finished cleanup task's exception unless the task was cancelled.

    Args:
        task: Local computation or cleanup task to cancel, await, or release.

    Returns:
        None: No return value.
    """
    if not task.cancelled():
        task.exception()


def _release_cleanup(task: asyncio.Future[None]) -> None:
    """Drop a finished retained cleanup task and consume its exception.

    Args:
        task: Local computation or cleanup task to cancel, await, or release.

    Returns:
        None: No return value.
    """
    _pending_cleanup.discard(task)
    _consume_exception(task)
