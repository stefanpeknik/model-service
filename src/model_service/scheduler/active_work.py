"""Track live tasks in one scheduler process for prompt cancellation.

Database leases remain authoritative. This registry is empty after a restart and cannot
cancel tasks owned by another process.
"""

from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import asyncio

    from model_service.scheduler.domain import Settlement


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
        """Associate an execution with its current attempt, replacing any previous local entry."""
        self._tasks[execution_id] = (fencing_token, task)

    def unregister(self, execution_id: str, fencing_token: str) -> None:
        """Remove an entry only if its token matches, so old cleanup cannot remove a newer attempt."""
        current = self._tasks.get(execution_id)
        if current is not None and current[0] == fencing_token:
            self._tasks.pop(execution_id)

    def cancel(self, execution_id: str) -> None:
        """Request asyncio cancellation of the local task, if present. Remote work may still be running."""
        current = self._tasks.get(execution_id)
        if current is not None:
            current[1].cancel()
