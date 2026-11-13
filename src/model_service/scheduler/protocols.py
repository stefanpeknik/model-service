"""What the API's and providers' routes need from the scheduler.

ExecutionScheduler implements it, and documents each method; tests pass fakes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from model_service.scheduler.public import (
        ExecutionView,
        ScheduleRequest,
        ScheduleResult,
    )


class Scheduler(Protocol):
    @property
    def alive(self) -> bool:
        """Whether the scheduler's worker is still running."""
        ...

    async def check_ready(self) -> None:
        """Check that durable storage is reachable."""
        ...

    async def schedule(self, request: ScheduleRequest) -> ScheduleResult:
        """Admit a durable builder execution, or return the one its idempotency key names."""
        ...

    async def get(self, execution_id: str) -> ExecutionView:
        """Read one execution."""
        ...

    async def get_many(self, execution_ids: list[str]) -> list[ExecutionView]:
        """Read the listed executions that exist."""
        ...

    async def cancel(self, execution_id: str) -> ExecutionView:
        """Cancel an execution."""
        ...
