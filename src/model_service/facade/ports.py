"""What the public routes need from the rest of the facade process.

The routes depend on these interfaces; the synchronous builder service
implements them. Kept apart from the routes so the implementations do not
import the HTTP layer.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from model_service.builders.contracts import BuilderResult
    from model_service.facade.requests import ImmediateBuilderRequest


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
