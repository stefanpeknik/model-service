"""Specify the EMPAIA job boundary, protected credentials, and configurable gateway factory.

Provider admission and finish use this boundary in the scheduler process; input access
uses it in the builder process.
"""

from __future__ import annotations

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

    @classmethod
    def invalid(cls, safe_message: str) -> EmpaiaGatewayError:
        """Rejected or malformed integration data: retrying will not help."""
        return cls(safe_message, retryable=False)

    @classmethod
    def unavailable(cls, safe_message: str) -> EmpaiaGatewayError:
        """A transient failure: the attempt may be retried."""
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

    async def get_mode(self) -> EmpaiaJobMode:
        """Fetch how the job was started."""
        ...

    async def has_input(self, name: str) -> bool:
        """Whether the job has the named input; optional builder inputs depend on the app's mode."""
        ...

    async def materialize_input(
        self,
        name: str,
        directory: str,
    ) -> str:
        """Make a named job input available inside `directory`; return its file path."""
        ...

    async def post_primitive(
        self, name: str, output_type: str, value: JsonValue
    ) -> None:
        """Post one output as a primitive of the given EMPAIA type."""
        ...

    async def slide_info(self) -> SlideInfo:
        """The job's slide with every level's downsample factor."""
        ...

    async def publish_pixelmap(
        self, name: str, slide_id: str, tile_sets: tuple[PixelmapTileSet, ...]
    ) -> None:
        """Create one output pixelmap on the slide and upload its levels."""
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
