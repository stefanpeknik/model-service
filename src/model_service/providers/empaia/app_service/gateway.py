"""Specify the EMPAIA job boundary, protected credentials, and configurable gateway factory."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Protocol

from pydantic import BaseModel, ConfigDict, Field


if TYPE_CHECKING:
    from model_service.contracts.json_values import JsonValue


type MutationGuard = Callable[[], Awaitable[None]]


class PrimitiveType(StrEnum):
    """An EMPAIA primitive output type."""

    BOOL = "bool"
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"


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
            safe_message: A message safe to log and show the job's user.
            retryable: Whether retrying the attempt may succeed.
            outcome_unknown: Whether a remote mutation may have been applied.
        """
        super().__init__(safe_message)
        self.safe_message = safe_message
        self.retryable = retryable
        self.outcome_unknown = outcome_unknown

    @classmethod
    def invalid(cls, safe_message: str) -> EmpaiaGatewayError:
        """Create a non-retryable error for rejected or malformed integration data."""
        return cls(safe_message, retryable=False)


class EmpaiaJobMode(StrEnum):
    """How the EMPAIA job was started, as AppService reports it."""

    STANDALONE = "STANDALONE"
    PREPROCESSING = "PREPROCESSING"
    POSTPROCESSING = "POSTPROCESSING"
    REPORT = "REPORT"


class EmpaiaJobContext(BaseModel):
    """Carry the API origin and job ID with the private job credential; the credential is hidden from repr."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    api_url: Annotated[str, Field(min_length=1)]
    job_id: Annotated[str, Field(min_length=1)]
    credential: Annotated[str, Field(min_length=1, repr=False)]


class EmpaiaGateway(ABC):
    """Abstract AppService configuration, input acquisition, result publication, and terminal reporting."""

    @abstractmethod
    def set_before_mutation(self, callback: MutationGuard) -> None:
        """Install an async guard to persist the scheduler's mutation marker before external writes."""
        ...

    @abstractmethod
    async def get_configuration(self) -> dict[str, JsonValue]:
        """Fetch the job's resolved provider configuration as JSON values."""
        ...

    @abstractmethod
    async def get_mode(self) -> EmpaiaJobMode:
        """Fetch how the job was started."""
        ...

    @abstractmethod
    async def has_input(self, name: str) -> bool:
        """Check whether the job has the named input; optional builder inputs depend on the app's mode."""
        ...

    @abstractmethod
    async def materialize_input(
        self,
        name: str,
        directory: str,
    ) -> str:
        """Make a named job input available inside `directory`."""
        ...

    @abstractmethod
    async def post_primitive(
        self, name: str, output_type: str, value: JsonValue
    ) -> None:
        """Post one output as a primitive of the given EMPAIA type."""
        ...

    @abstractmethod
    async def finalize(self) -> None:
        """Mark the external job's successful completion after its outputs are published."""
        ...

    @abstractmethod
    async def report_failure(self, message: str) -> None:
        """Report a safe failure or cancellation message to the external job."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release the gateway's HTTP resources."""
        ...


class EmpaiaGatewayFactory(Protocol):
    """Construct an invocation-scoped gateway from protected job connection settings."""

    def __call__(self, context: EmpaiaJobContext) -> EmpaiaGateway:
        """Create a gateway for one external job using its origin, ID, and private credential.

        Args:
            context: The job's connection settings.

        Returns:
            The gateway; the caller closes it.

        Raises:
            EmpaiaGatewayError: If the connection settings are incomplete.
        """
        ...
