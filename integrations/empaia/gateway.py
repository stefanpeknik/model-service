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
    from workflow_execution.contracts import JsonValue, WorkflowOutput


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
        """Capture the safe message and outcome classification without embedding credentials in error text.

        Args:
            safe_message: Message permitted in public errors without internal details or
                credentials.
            retryable: Whether another attempt is permitted for this classified failure.
            outcome_unknown: Whether an external mutation may have occurred despite
                missing confirmation.
        """
        super().__init__(safe_message)
        self.safe_message = safe_message
        self.retryable = retryable
        self.outcome_unknown = outcome_unknown


class EmpaiaInputState(BaseModel):
    """Carry the non-secret API origin and job ID used to select provider input access.

    Attributes:
        api_url: External AppService origin.
        job_id: External EMPAIA job identifier.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    api_url: Annotated[str, Field(min_length=1)]
    job_id: Annotated[str, Field(min_length=1)]


class EmpaiaJobContext(EmpaiaInputState):
    """Add a private job credential to the public job identity; the credential is hidden from repr.

    Attributes:
        credential: Private bearer credential hidden from representations.
    """

    credential: Annotated[str, Field(min_length=1, repr=False)]

    def protected_values(self) -> dict[str, JsonValue]:
        """Extract API origin, job ID, and credential for encrypted durable context and internal dispatch.

        Returns:
            dict[str, JsonValue]: Job origin, ID, and private credential for protected
                storage/internal dispatch.
        """
        return {
            "api_url": self.api_url,
            "job_id": self.job_id,
            "credential": self.credential,
        }


class EmpaiaGateway(Protocol):
    """Abstract AppService configuration, input acquisition, result publication, and terminal reporting."""

    def set_before_mutation(self, callback: MutationGuard) -> None:
        """Install an async guard to persist the scheduler's mutation marker before external writes.

        Args:
            callback: Async callback invoked before an external provider mutation.

        Returns:
            None: No return value.
        """
        ...

    async def get_configuration(self) -> dict[str, JsonValue]:
        """Fetch the job's resolved provider configuration as JSON values.

        Returns:
            dict[str, JsonValue]: Merged provider configuration for the current external
                job.

        Raises:
            EmpaiaGatewayError: AppService transport, authentication, response data, or
                image conversion fails.
        """
        ...

    async def materialize_input(self, name: str, destination: str) -> None:
        """Make a named job input available at the requested local destination.

        Args:
            name: Declared workflow input name to resolve.
            destination: Local path where input materialization or mask writing must
                produce a file.

        Returns:
            None: No return value.

        Raises:
            EmpaiaGatewayError: AppService transport, authentication, response data, or
                image conversion fails.
        """
        ...

    async def publish_output(self, name: str, output: WorkflowOutput) -> None:
        """Publish one inline value or completed shared artifact to the external job.

        Args:
            name: Declared workflow output name registered with AppService.
            output: Inline JSON value or completed shared artifact to publish.

        Returns:
            None: No return value.

        Raises:
            EmpaiaGatewayError: AppService transport, authentication, response data, or
                image conversion fails.
        """
        ...

    async def finalize(self) -> None:
        """Mark the external job's successful completion after its outputs are published.

        Returns:
            None: No return value.

        Raises:
            EmpaiaGatewayError: AppService transport, authentication, response data, or
                image conversion fails.
        """
        ...

    async def report_failure(self, message: str) -> None:
        """Report a safe failure or cancellation message to the external job.

        Args:
            message: Safe user-facing failure or cancellation text.

        Returns:
            None: No return value.

        Raises:
            EmpaiaGatewayError: AppService transport, authentication, response data, or
                image conversion fails.
        """
        ...

    async def close(self) -> None:
        """Release the gateway's HTTP resources after admission, input use, or reporting.

        Returns:
            None: No return value.

        Raises:
            EmpaiaGatewayError: AppService transport, authentication, response data, or
                image conversion fails.
        """
        ...


class EmpaiaGatewayFactory(Protocol):
    """Construct an invocation-scoped gateway from protected job connection settings."""

    def __call__(self, context: EmpaiaJobContext) -> EmpaiaGateway:
        """Create a gateway for one external job using its origin, ID, and private credential.

        Args:
            context: External job origin/ID and protected bearer credential.

        Returns:
            EmpaiaGateway: Job-scoped provider communication boundary.
        """
        ...


def load_gateway_factory(import_path: str) -> EmpaiaGatewayFactory:
    """Import a callable configured as module:attribute, rejecting invalid syntax or a noncallable target.

    Args:
        import_path: Configured factory location in module:attribute syntax.

    Returns:
        EmpaiaGatewayFactory: Imported callable that constructs a job-scoped gateway.

    Raises:
        ValueError: import_path does not use a nonempty module:attribute pair.
        ImportError: The configured module cannot be imported.
        TypeError: The configured attribute is absent or is not callable.
    """
    module_name, separator, attribute_name = import_path.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("EMPAIA gateway factory must use module:attribute syntax")
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute_name, None)
    if not callable(factory):
        raise TypeError("EMPAIA gateway factory is not callable")
    return factory
