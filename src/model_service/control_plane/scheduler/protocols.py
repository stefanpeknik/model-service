"""What the control plane's and providers' routes need from the scheduler, and what it runs analyses through.

ExecutionScheduler implements `Scheduler` and HTTPAnalysisDispatch implements
`AnalysisDispatch`; each documents its methods. Tests pass fakes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from model_service.contracts.analysis_http import AnalysisRunRequest
    from model_service.contracts.result import RunResult
    from model_service.control_plane.scheduler.public import (
        AnalysisScheduleRequest,
        ExecutionView,
        ScheduleResult,
    )


class Scheduler(Protocol):
    """What the control plane needs from the execution scheduler."""

    @property
    def alive(self) -> bool:
        """Whether the scheduler's worker is still running."""
        ...

    async def check_ready(self) -> None:
        """Check that durable storage is reachable."""
        ...

    async def schedule(self, request: AnalysisScheduleRequest) -> ScheduleResult:
        """Admit a durable analysis execution, or return the one its idempotency key names.

        Args:
            request: The schedule request.

        Returns:
            The execution and whether this request created it.
        """
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


class AnalysisDispatch(Protocol):
    """How the control plane runs a resolved analysis on the analysis runner."""

    async def run_analysis(self, request: AnalysisRunRequest) -> RunResult:
        """Run one resolved analysis and return its result.

        Args:
            request: The resolved analysis, invocation and execution context.

        Returns:
            The run's result.
        """
        ...
