from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    import asyncio

    from execution_scheduler.domain import Settlement


class ActiveWorkRegistry:
    def __init__(self) -> None:
        self._tasks: dict[str, tuple[str, asyncio.Task[Settlement]]] = {}

    def register(
        self,
        execution_id: str,
        fencing_token: str,
        task: asyncio.Task[Settlement],
    ) -> None:
        self._tasks[execution_id] = (fencing_token, task)

    def unregister(self, execution_id: str, fencing_token: str) -> None:
        current = self._tasks.get(execution_id)
        if current is not None and current[0] == fencing_token:
            self._tasks.pop(execution_id)

    def cancel(self, execution_id: str) -> None:
        current = self._tasks.get(execution_id)
        if current is not None:
            current[1].cancel()
