"""Specify the EMPAIA job boundary, protected credentials, and configurable gateway factory.

Provider admission and finish use this boundary in the scheduler process; input access
uses it in the workflow process.
"""

from __future__ import annotations

import importlib
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Annotated, Protocol

from pydantic import BaseModel, ConfigDict, Field


if TYPE_CHECKING:
    from model_service.builders.contracts import BuilderOutput, JsonValue


type MutationGuard = Callable[[], Awaitable[None]]


class EmpaiaGatewayError(RuntimeError):
    """Carry safe integration failure metadata, including retry permission and uncertain remote mutations."""

    def __init__(
        self,
        safe_message: str,
        *,
        retryable: bool,
        outcome_unknown: bool = False,
    ) -> None:
        """Capture the safe message and outcome classification without embedding credentials in error text."""
        super().__init__(safe_message)
        self.safe_message = safe_message
        self.retryable = retryable
        self.outcome_unknown = outcome_unknown


class EmpaiaInputState(BaseModel):
    """Carry the non-secret API origin and job ID used to select provider input access."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    api_url: Annotated[str, Field(min_length=1)]
    job_id: Annotated[str, Field(min_length=1)]


class EmpaiaJobContext(EmpaiaInputState):
    """Add a private job credential to the public job identity; the credential is hidden from repr."""

    credential: Annotated[str, Field(min_length=1, repr=False)]

    def protected_values(self) -> dict[str, JsonValue]:
        """Extract API origin, job ID, and credential for encrypted durable context and internal dispatch."""
        return {
            "api_url": self.api_url,
            "job_id": self.job_id,
            "credential": self.credential,
        }


class EmpaiaGateway(Protocol):
    """Abstract AppService configuration, input acquisition, result publication, and terminal reporting."""

    def set_before_mutation(self, callback: MutationGuard) -> None:
        """Install an async guard to persist the scheduler's mutation marker before external writes."""
        ...

    async def get_configuration(self) -> dict[str, JsonValue]:
        """Fetch the job's resolved provider configuration as JSON values."""
        ...

    async def materialize_input(
        self, name: str, directory: str, *, mpp: float | None = None
    ) -> str:
        """Make a named job input available inside `directory`; return its file path.

        `mpp` is the resolution the builder reads, for gateways that fetch slides.
        """
        ...

    async def publish_output(self, name: str, output: BuilderOutput) -> None:
        """Publish one inline value or completed shared artifact to the external job."""
        ...

    async def finalize(self) -> None:
        """Mark the external job's successful completion after its outputs are published."""
        ...

    async def report_failure(self, message: str) -> None:
        """Report a safe failure or cancellation message to the external job."""
        ...

    async def close(self) -> None:
        """Release the gateway's HTTP resources after admission, input use, or reporting."""
        ...


class EmpaiaGatewayFactory(Protocol):
    """Construct an invocation-scoped gateway from protected job connection settings."""

    def __call__(self, context: EmpaiaJobContext) -> EmpaiaGateway:
        """Create a gateway for one external job using its origin, ID, and private credential."""
        ...


def load_gateway_factory(import_path: str) -> EmpaiaGatewayFactory:
    """Import a callable configured as module:attribute, rejecting invalid syntax or a noncallable target."""
    module_name, separator, attribute_name = import_path.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("EMPAIA gateway factory must use module:attribute syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name, None)
    if not callable(factory):
        raise TypeError("EMPAIA gateway factory is not callable")
    return factory
