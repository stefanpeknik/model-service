"""What the API's and providers' routes need from the scheduler, and what it runs builders through.

ExecutionScheduler implements `Scheduler` and HTTPBuilderDispatch implements
`BuilderDispatch`; each documents its methods. Tests pass fakes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from model_service.api.scheduler.public import (
        ExecutionView,
        ScheduleRequest,
        ScheduleResult,
    )
    from model_service.contracts.invocation import ExecutionContext, Invocation
    from model_service.contracts.model_declarations import ModelBinding
    from model_service.contracts.result import BuilderResult


class Scheduler(Protocol):
    """What the API needs from the execution scheduler."""

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


class BuilderDispatch(Protocol):
    """How the scheduler runs a builder on a replica."""

    async def run(
        self,
        builder_id: str,
        invocation: Invocation,
        model: ModelBinding,
        execution_context: ExecutionContext,
    ) -> BuilderResult:
        """Run one builder invocation and return its result."""
        ...
