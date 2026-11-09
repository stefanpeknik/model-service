"""Specify the EMPAIA job boundary, protected credentials, and configurable gateway factory.

Provider admission and finish use this boundary in the scheduler process; input access
uses it in the builder process.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Protocol

from pydantic import BaseModel, ConfigDict, Field


if TYPE_CHECKING:
    from rationai.empaia.typing import SlideInfo
    from rationai.empaia.utils.scalar_output import PixelmapTileSet

    from model_service.builders.contracts import JsonValue


type MutationGuard = Callable[[], Awaitable[None]]

PROVIDER_NAME = "empaia"
INPUT_ACCESS_KIND = "empaia"


class PrimitiveType(StrEnum):
    """An EMPAIA primitive output type."""

    BOOL = "bool"
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"

    def fits(self, value: object) -> bool:
        """Whether EMPAIA takes this value as a primitive of this type.

        A bool is an int in Python, so it is told apart; a float takes whole
        numbers; an empty string is refused.

        Args:
            value: The output's value.

        Returns:
            Whether the value fits.
        """
        if isinstance(value, bool):
            return self is PrimitiveType.BOOL
        match self:
            case PrimitiveType.INTEGER:
                return isinstance(value, int)
            case PrimitiveType.FLOAT:
                return isinstance(value, int | float)
            case PrimitiveType.STRING:
                return isinstance(value, str) and bool(value)
        return False


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
        """Create an error for rejected or malformed integration data; retrying will not help.

        Args:
            safe_message: A message safe to log and show the job's user.

        Returns:
            A non-retryable error with a known outcome.
        """
        return cls(safe_message, retryable=False)

    @classmethod
    def unavailable(cls, safe_message: str) -> EmpaiaGatewayError:
        """Create an error for a transient failure; the attempt may be retried.

        Args:
            safe_message: A message safe to log and show the job's user.

        Returns:
            A retryable error with a known outcome.
        """
        return cls(safe_message, retryable=True)


class EmpaiaJobMode(StrEnum):
    """How the EMPAIA job was started, as AppService reports it."""

    STANDALONE = "STANDALONE"
    PREPROCESSING = "PREPROCESSING"
    POSTPROCESSING = "POSTPROCESSING"
    REPORT = "REPORT"


class EmpaiaInputState(BaseModel):
    """Carry the non-secret API origin and job ID used to select provider input access."""

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    api_url: Annotated[str, Field(min_length=1)]
    job_id: Annotated[str, Field(min_length=1)]
    # Recorded at admission; nothing depends on it yet. Executions admitted
    # before it was recorded have none.
    mode: Annotated[EmpaiaJobMode | None, Field(strict=False)] = None


class EmpaiaJobContext(EmpaiaInputState):
    """Add a private job credential to the public job identity; the credential is hidden from repr."""

    credential: Annotated[str, Field(min_length=1, repr=False)]

    def protected_values(self) -> dict[str, JsonValue]:
        """Extract API origin, job ID, and credential for encrypted durable context and internal dispatch.

        Returns:
            The values by field name, credential included.
        """
        return {
            "api_url": self.api_url,
            "job_id": self.job_id,
            "credential": self.credential,
        }


class EmpaiaGateway(ABC):
    """Abstract AppService configuration, input acquisition, result publication, and terminal reporting."""

    @abstractmethod
    def set_before_mutation(self, callback: MutationGuard) -> None:
        """Install an async guard to persist the scheduler's mutation marker before external writes.

        Args:
            callback: Awaited before each external write.
        """
        ...

    @abstractmethod
    async def get_configuration(self) -> dict[str, JsonValue]:
        """Fetch the job's resolved provider configuration as JSON values.

        Returns:
            The configuration, by key.

        Raises:
            EmpaiaGatewayError: If the request fails or the configuration is invalid.
        """
        ...

    @abstractmethod
    async def get_mode(self) -> EmpaiaJobMode:
        """Fetch how the job was started.

        Returns:
            The job's mode.

        Raises:
            EmpaiaGatewayError: If the request fails or the mode is unknown.
        """
        ...

    @abstractmethod
    async def has_input(self, name: str) -> bool:
        """Check whether the job has the named input; optional builder inputs depend on the app's mode.

        Args:
            name: The EMPAIA input key.

        Returns:
            True if the job has the input.

        Raises:
            EmpaiaGatewayError: If the request fails other than by a missing input.
        """
        ...

    @abstractmethod
    async def materialize_input(
        self,
        name: str,
        directory: str,
    ) -> str:
        """Make a named job input available inside `directory`.

        Args:
            name: The EMPAIA input key.
            directory: The directory to put the file in.

        Returns:
            The path of the file.

        Raises:
            EmpaiaGatewayError: If the input cannot be fetched or materialized.
        """
        ...

    @abstractmethod
    async def post_primitive(
        self, name: str, output_type: str, value: JsonValue
    ) -> None:
        """Post one output as a primitive of the given EMPAIA type.

        Args:
            name: The EMPAIA output key.
            output_type: The EMPAIA primitive type, e.g. `integer`.
            value: The output's value.

        Raises:
            EmpaiaGatewayError: If the value is not a valid primitive or the write
                fails; a failed write may leave an unknown outcome.
        """
        ...

    @abstractmethod
    async def slide_info(self) -> SlideInfo:
        """Fetch the job's slide with every level's downsample factor.

        Returns:
            The slide.

        Raises:
            EmpaiaGatewayError: If the request fails.
        """
        ...

    @abstractmethod
    async def publish_pixelmap(
        self, name: str, slide_id: str, tile_sets: tuple[PixelmapTileSet, ...]
    ) -> None:
        """Create one output pixelmap on the slide and upload its levels.

        Args:
            name: The EMPAIA output key, also the pixelmap's name.
            slide_id: The slide the pixelmap belongs to.
            tile_sets: The pyramid levels to upload.

        Raises:
            EmpaiaGatewayError: If creating or uploading fails; a failed write may
                leave an unknown outcome.
        """
        ...

    @abstractmethod
    async def finalize(self) -> None:
        """Mark the external job's successful completion after its outputs are published.

        Raises:
            EmpaiaGatewayError: If the request fails; the outcome may be unknown.
        """
        ...

    @abstractmethod
    async def report_failure(self, message: str) -> None:
        """Report a safe failure or cancellation message to the external job.

        Args:
            message: The safe message to report.

        Raises:
            EmpaiaGatewayError: If the request fails; the outcome may be unknown.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release the gateway's HTTP resources after admission, input use, or reporting."""
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
