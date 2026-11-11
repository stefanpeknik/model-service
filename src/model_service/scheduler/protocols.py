"""What the API needs from the scheduler: the Scheduler interface.

The API's routes and providers' routes depend on these; ExecutionScheduler
implements Scheduler.
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
        """Check that durable storage is reachable.

        Raises:
            SchedulingUnavailableError: If the execution repository is unavailable.
        """
        ...

    async def schedule(self, request: ScheduleRequest) -> ScheduleResult:
        """Admit a durable builder execution, or return the one its idempotency key names.

        Args:
            request: The builder, model, provider, body and idempotency key.

        Returns:
            The execution and whether it was created.

        Raises:
            UnknownBuilderError: If the builder is not registered.
            UnknownModelError: If the model is not deployed.
            IncompatibleModelError: If the builder does not accept the model.
            ScheduleConflictError: If the idempotency key names a different request.
            InvalidScheduleRequestError: If the request is invalid.
            SchedulingUnavailableError: If the execution repository is unavailable.
        """
        ...

    async def get(self, execution_id: str) -> ExecutionView:
        """Read one execution.

        Args:
            execution_id: The execution ID.

        Returns:
            The execution's public view.

        Raises:
            UnknownExecutionError: If no execution has this ID.
            SchedulingUnavailableError: If the execution repository is unavailable.
        """
        ...

    async def get_many(self, execution_ids: list[str]) -> list[ExecutionView]:
        """Read the listed executions that exist.

        Args:
            execution_ids: The execution IDs.

        Returns:
            Views of the existing executions; unknown IDs are left out.

        Raises:
            SchedulingUnavailableError: If the execution repository is unavailable.
        """
        ...

    async def cancel(self, execution_id: str) -> ExecutionView:
        """Cancel an execution.

        Args:
            execution_id: The execution ID.

        Returns:
            The execution's view after the cancellation request.

        Raises:
            UnknownExecutionError: If no execution has this ID.
            SchedulingUnavailableError: If the execution repository is unavailable.
        """
        ...
