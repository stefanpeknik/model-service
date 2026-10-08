"""Track live tasks in one scheduler process for prompt cancellation.

Database leases remain authoritative. This registry is empty after a restart and cannot
cancel tasks owned by another process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import asyncio

    from execution_scheduler.domain import Settlement


class ActiveWorkRegistry:
    """Map execution IDs to local tasks and the fencing tokens of their current attempts."""

    def __init__(self) -> None:
        """Start with no locally owned tasks; construction creates no background work."""
        self._tasks: dict[str, tuple[str, asyncio.Task[Settlement]]] = {}

    def register(
        self,
        execution_id: str,
        fencing_token: str,
        task: asyncio.Task[Settlement],
    ) -> None:
        """Associate an execution with its current attempt, replacing any previous local entry.

        Args:
            execution_id: Durable execution identifier, not an external provider job ID.
            fencing_token: Current claim token; stale tokens cannot mutate execution
                state.
            task: Local computation or cleanup task to cancel, await, or release.

        Returns:
            None: No return value.
        """
        self._tasks[execution_id] = (fencing_token, task)

    def unregister(self, execution_id: str, fencing_token: str) -> None:
        """Remove an entry only if its token matches, so old cleanup cannot remove a newer attempt.

        Args:
            execution_id: Durable execution identifier, not an external provider job ID.
            fencing_token: Current claim token; stale tokens cannot mutate execution
                state.

        Returns:
            None: No return value.
        """
        current = self._tasks.get(execution_id)
        if current is not None and current[0] == fencing_token:
            self._tasks.pop(execution_id)

    def cancel(self, execution_id: str) -> None:
        """Request asyncio cancellation of the local task, if present. Remote work may still be running.

        Args:
            execution_id: Durable execution identifier, not an external provider job ID.

        Returns:
            None: No return value.
        """
        current = self._tasks.get(execution_id)
        if current is not None:
            current[1].cancel()
