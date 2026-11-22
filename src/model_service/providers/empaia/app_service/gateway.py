"""Specify the EMPAIA job boundary, protected credentials, and configurable gateway factory.

Provider admission and finish use this boundary in the scheduler process; input access
uses it in the analysis runner.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from model_service.contracts.json_values import JsonValue


if TYPE_CHECKING:
    from rationai.empaia.typing import SlideInfo
    from rationai.empaia.utils.scalar_output import PixelmapTileSet

    from model_service.providers.empaia.app_service.types import DirectAccess


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
        """Create a non-retryable error for rejected or malformed integration data."""
        return cls(safe_message, retryable=False)


class EmpaiaJobMode(StrEnum):
    """How the EMPAIA job was started, as AppService reports it."""

    STANDALONE = "STANDALONE"
    PREPROCESSING = "PREPROCESSING"
    POSTPROCESSING = "POSTPROCESSING"
    REPORT = "REPORT"


type ComponentUse = dict[str, JsonValue]
"""A converter or publisher as recorded at admission: `{"type": ..., **options}`."""


class EmpaiaInputState(BaseModel):
    """Carry the non-secret API origin and job ID used to select provider input access.

    Attributes:
        api_url: The job's App Service URL.
        job_id: The EMPAIA job ID.
        mode: How the job was started; None for executions admitted before
            it was recorded.
        inputs: Each file input's EAD type and converter options, by input
            name, as admitted; None for builder-era jobs, which no longer
            run.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    api_url: Annotated[str, Field(min_length=1)]
    job_id: Annotated[str, Field(min_length=1)]
    # Recorded at admission; nothing depends on it yet. Executions admitted
    # before it was recorded have none.
    mode: Annotated[EmpaiaJobMode | None, Field(strict=False)] = None
    inputs: dict[str, ComponentUse] | None = None


class Publications(BaseModel):
    """What admission fixed about publishing an analysis's outputs.

    Attributes:
        slide: The app's slide input, which outputs reference.
        outputs: Each published output's EAD type and publisher options, by
            output name, in publication order.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    slide: Annotated[str, Field(min_length=1)]
    outputs: dict[str, ComponentUse]


class EmpaiaJobContext(EmpaiaInputState):
    """Add a private job credential to the public job identity; the credential is hidden from repr.

    Attributes:
        credential: The job's App Service token.
        publications: How the analysis's outputs are published, as
            admitted; None for a builder-era job, which is never
            published now.
    """

    credential: Annotated[str, Field(min_length=1, repr=False)]
    publications: Publications | None = None

    def protected_values(self) -> dict[str, JsonValue]:
        """Extract API origin, job ID, credential and publications for encrypted durable context.

        Returns:
            The values by field name, credential included; publications only
            when set.
        """
        values: dict[str, JsonValue] = {
            "api_url": self.api_url,
            "job_id": self.job_id,
            "credential": self.credential,
        }
        if self.publications is not None:
            values["publications"] = self.publications.model_dump(mode="json")
        return values


class EmpaiaGateway(ABC):
    """The App Service calls of one job, one method per SDK call family.

    Configuration and mode; input reads (presence, raw input data, the
    slide-volume path, the typed slide); posting an output body; uploading a
    pixelmap pyramid; progress, finalization and failure. Converters and
    publishers build on these, so a new EMPAIA type needs no new method.
    """

    @property
    @abstractmethod
    def job_id(self) -> str:
        """The EMPAIA job, which creates the outputs."""
        ...

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
        """Check whether the job has the named input; optional inputs depend on the app's mode."""
        ...

    @abstractmethod
    async def get_input_data(self, name: str) -> JsonValue:
        """Fetch the job's named input as the App Service gives it.

        Args:
            name: The EMPAIA input key.

        Returns:
            The decoded input data, e.g. a primitive or a rectangle.
        """
        ...

    @abstractmethod
    async def direct_access(self, name: str) -> DirectAccess | None:
        """Fetch where the named input lies on the slide volume.

        Args:
            name: The EMPAIA input key.

        Returns:
            The path and whether direct access is supported, or None if the
            input has no path there.
        """
        ...

    @abstractmethod
    async def slide_info(self, name: str) -> SlideInfo:
        """Fetch the job's named slide with every level's downsample factor.

        Args:
            name: The EMPAIA input key of the slide.

        Returns:
            The slide, as the SDK types it.
        """
        ...

    @abstractmethod
    async def post_output(self, name: str, body: Any) -> None:
        """Post one output's body, e.g. a primitive.

        Args:
            name: The EMPAIA output key.
            body: The SDK's request model of the output.
        """
        ...

    @abstractmethod
    async def publish_pixelmap(
        self, name: str, slide_id: str, tile_sets: tuple[PixelmapTileSet, ...]
    ) -> None:
        """Create one output pixelmap on the slide and upload its levels."""
        ...

    @abstractmethod
    async def finalize(self) -> None:
        """Mark the external job's successful completion after its outputs are published."""
        ...

    @abstractmethod
    async def report_progress(self, fraction: float) -> None:
        """Report how far the job got, which the viewer shows while it runs.

        Args:
            fraction: Share of the work done, from 0 to 1.
        """
        ...

    @abstractmethod
    async def report_failure(self, message: str) -> None:
        """Report a safe failure or cancellation message to the external job."""
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
