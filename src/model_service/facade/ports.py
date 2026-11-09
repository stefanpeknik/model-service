"""What the public routes need from the rest of the facade process.

The routes depend on these interfaces; the scheduler, the capacity explainer
and the synchronous builder service implement them. Kept apart from the routes
so the implementations do not import the HTTP layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from model_service.builders.contracts import BuilderResult
    from model_service.facade.requests import ImmediateBuilderRequest
    from model_service.scheduler.contracts import (
        ExecutionView,
        InferenceResult,
        InferenceScheduleRequest,
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

    async def schedule_inference(
        self, request: InferenceScheduleRequest
    ) -> ScheduleResult:
        """Admit a durable raw inference execution, or return the one its key names.

        Args:
            request: The model, payload, headers and idempotency key.

        Returns:
            The execution and whether it was created.

        Raises:
            UnknownModelError: If the model is not deployed.
            IncompatibleInputError: If the payload is not a valid tile for the model.
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

    async def inference_result(
        self, execution_id: str
    ) -> tuple[InferenceResult, bytes]:
        """Read the stored model response of a succeeded inference execution.

        Args:
            execution_id: The execution ID.

        Returns:
            The response's media type and headers, and its body.

        Raises:
            UnknownExecutionError: If no execution has this ID.
            ResultUnavailableError: If the execution has no stored response.
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


class Capacity(ABC):
    @abstractmethod
    async def explain(self, execution: ExecutionView) -> ExecutionView:
        """Add a message explaining why a running execution makes no progress.

        Args:
            execution: The execution to explain.

        Returns:
            The execution, with a message if one applies.
        """
        ...


class NoCapacityInfo(Capacity):
    async def explain(self, execution: ExecutionView) -> ExecutionView:
        """Leave the execution unexplained, for deployments without a Ray dashboard.

        Args:
            execution: The execution.

        Returns:
            The execution unchanged.
        """
        return execution


class BuilderRunner(ABC):
    @abstractmethod
    async def run(
        self, builder_id: str, model_id: str, request: ImmediateBuilderRequest
    ) -> BuilderResult:
        """Run a builder with a model while the caller waits.

        Args:
            builder_id: The builder to run.
            model_id: The model the builder uses.
            request: Inputs, configuration and deadline.

        Returns:
            The builder's result.

        Raises:
            UnknownBuilderError: If the builder is not registered.
            UnknownModelError: If the model is not deployed.
            IncompatibleModelError: If the builder does not accept the model.
            InvalidInvocation: If the request is invalid.
            ConfirmedRejectionError: If the builder refused for lack of capacity.
            RemoteFailureError: If the builder failed.
            UnknownOutcomeError: If it is unknown whether the builder ran.
        """
        ...
