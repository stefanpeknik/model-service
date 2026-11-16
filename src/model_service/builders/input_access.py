"""How builders reach their inputs: the interface providers implement, and closing it.

A provider's access adapter turns an invocation's provider-specific inputs into
local files for one run, and says where outputs go; builders see only the
resulting paths. close_resource closes an access, or anything it holds, within
a timeout; cleanup tasks stay referenced until they finish and their exceptions
are consumed, so cancelling the caller never leaks a half-closed resource.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    from model_service.contracts.invocation import ExecutionContext, Invocation
    from model_service.contracts.model_declarations import ModelBinding


class InputAccess(ABC):
    """Resolves one invocation's inputs and outputs to local files."""

    @abstractmethod
    def input_path(self, name: str) -> Path:
        """Resolve a named invocation input to a local file.

        Args:
            name: Input name from the builder's declaration.

        Returns:
            Local path the builder can read the input from.
        """
        ...

    @abstractmethod
    def output_path(self, relative_path: str) -> Path:
        """Resolve an output's relative path to the local file to write.

        Args:
            relative_path: Output path relative to the invocation's output location.

        Returns:
            Local path the builder writes the artifact to.
        """
        ...

    @abstractmethod
    def is_visible_artifact(self, path: Path) -> bool:
        """Check that a returned artifact is a file this access exposes to callers.

        Args:
            path: Artifact path a builder returned in its result.

        Returns:
            True when the path is a file inside the access's visible output area.
        """
        ...

    @abstractmethod
    async def aclose(self) -> None:
        """Release what the access holds, such as fetched inputs or temporary files."""
        ...


class InputAccessFactory(ABC):
    """Creates the input access of one input access kind, per invocation attempt."""

    # The model tells an adapter that fetches slides which resolution the builder reads.
    @abstractmethod
    def create(
        self,
        invocation: Invocation,
        execution_context: ExecutionContext,
        model: ModelBinding,
    ) -> InputAccess | Awaitable[InputAccess]:
        """Create the input access for one invocation attempt.

        Args:
            invocation: Invocation whose inputs and outputs the access resolves.
            execution_context: Attempt identity and protected provider state.
            model: Model binding; tells an adapter that fetches slides which
                resolution the builder reads.

        Returns:
            The access, or an awaitable resolving to it.
        """
        ...


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
        raise TimeoutError("builder resource cleanup timed out")
    if cancellation is not None:
        _consume_exception(task)
        raise cancellation
    if task.cancelled():
        raise RuntimeError("builder resource cleanup was cancelled")
    task.result()


# Keep cancellation-resistant cleanup alive until it actually finishes.
_pending_cleanup: set[asyncio.Future[None]] = set()


def _consume_exception(task: asyncio.Future[None]) -> None:
    """Retrieve a finished cleanup task's exception unless the task was cancelled."""
    if not task.cancelled():
        task.exception()


def _release_cleanup(task: asyncio.Future[None]) -> None:
    """Drop a finished retained cleanup task and consume its exception."""
    _pending_cleanup.discard(task)
    _consume_exception(task)
