"""Builder invocations, model bindings, results and safe errors.

Everything here crosses a process boundary (facade → Ray builder replica), so
the types are strict and JSON-serializable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, ClassVar, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    TypeAdapter,
    ValidationError,
)
from pydantic import JsonValue as PydanticJsonValue


if TYPE_CHECKING:
    from pathlib import Path


_JSON_VALUE: TypeAdapter[PydanticJsonValue] = TypeAdapter(
    PydanticJsonValue, config=ConfigDict(allow_inf_nan=False)
)


def _validate_json_value(value: object) -> PydanticJsonValue:
    """Validate a value as strict JSON, rejecting NaN and infinities.

    Args:
        value: Value to validate.

    Returns:
        The validated value.

    Raises:
        pydantic.ValidationError: The value is not strict JSON.
    """
    return _JSON_VALUE.validate_python(value, strict=True)


type JsonValue = Annotated[PydanticJsonValue, BeforeValidator(_validate_json_value)]
type _NonEmptyString = Annotated[str, Field(strict=True, min_length=1)]
type _JsonObject = Annotated[dict[str, JsonValue], Field(strict=True)]


def is_json_value(value: object) -> bool:
    """Check whether a value is strict JSON, without NaN or infinities.

    Args:
        value: Value to check.

    Returns:
        True when the value is strict JSON.
    """
    try:
        _JSON_VALUE.validate_python(value, strict=True)
    except ValidationError:
        return False
    return True


class InferenceContract(StrEnum):
    """The model-ready input and prediction structure a model deployment declares."""

    SPATIAL_SCORES = "spatial-scores"
    TILE_EMBEDDING = "tile-embedding"


class _ModelBinding(BaseModel):
    """A model's catalog declaration: what builders need to drive it.

    A model file gives each field in its `inference` block if the field is in
    `inference_fields`, otherwise in its first deployment's `user_config`, which
    the model implementation also reads.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    inference_fields: ClassVar[tuple[str, ...]] = ()

    id: Annotated[str, Field(min_length=1)]
    tile_size: Annotated[int, Field(strict=True, gt=0)]
    mpp: Annotated[float, Field(gt=0, allow_inf_nan=False)]


class SpatialScoresModel(_ModelBinding):
    """Ray-handle model returning `n_channels x output_tile_size²` scores in [0, 1]."""

    contract: Literal[InferenceContract.SPATIAL_SCORES] = (
        InferenceContract.SPATIAL_SCORES
    )
    output_tile_size: Annotated[int, Field(strict=True, gt=0)]
    n_channels: Annotated[int, Field(strict=True, gt=0)]


class TileEmbeddingModel(_ModelBinding):
    """Ray-handle model returning float32 embeddings for an RGB tile."""

    # The implementation hard-codes these; the file states them for the catalog.
    inference_fields = ("mpp",)

    contract: Literal[InferenceContract.TILE_EMBEDDING] = (
        InferenceContract.TILE_EMBEDDING
    )


type ModelBinding = Annotated[
    SpatialScoresModel | TileEmbeddingModel, Field(discriminator="contract")
]
MODEL_BINDING: TypeAdapter[ModelBinding] = TypeAdapter(ModelBinding)
# The declaration class of each contract.
MODEL_CLASSES: dict[
    InferenceContract, type[SpatialScoresModel | TileEmbeddingModel]
] = {
    model.model_fields["contract"].default: model
    for model in (SpatialScoresModel, TileEmbeddingModel)
}


class InputKind(StrEnum):
    SHARED_PATH = "shared_path"
    PROVIDER = "provider"


class OutputKind(StrEnum):
    INLINE = "inline"
    SHARED_ARTIFACT = "shared_artifact"


@dataclass(frozen=True, slots=True)
class SharedPathInput:
    """A file below the invocation's shared-storage root."""

    path: _NonEmptyString
    kind: Literal[InputKind.SHARED_PATH] = InputKind.SHARED_PATH


@dataclass(frozen=True, slots=True)
class ProviderInput:
    """An input the provider's access adapter materializes, e.g. from an external job."""

    name: _NonEmptyString
    kind: Literal[InputKind.PROVIDER] = InputKind.PROVIDER


type BuilderInput = Annotated[
    SharedPathInput | ProviderInput, Field(discriminator="kind")
]


# Which access adapter resolves an invocation's inputs and outputs; each
# provider names its own, the builder replica holds one adapter per kind.
type InputAccessKind = str
SHARED_STORAGE: InputAccessKind = "shared_storage"


@dataclass(frozen=True, slots=True)
class InputAccessContext:
    """Selects an access adapter and its public state."""

    kind: InputAccessKind
    state: _JsonObject


def _iso_datetime(value: object) -> datetime:
    """Parse a deadline from an ISO 8601 string, passing datetimes through.

    Args:
        value: Datetime or ISO 8601 string.

    Returns:
        The parsed datetime.

    Raises:
        ValueError: The value is neither a datetime nor a valid ISO 8601 string.
    """
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        raise ValueError("deadline_at must be an ISO 8601 string")
    return datetime.fromisoformat(value)


type Deadline = Annotated[
    datetime,
    BeforeValidator(_iso_datetime),
    PlainSerializer(datetime.isoformat, return_type=str),
]


@dataclass(frozen=True, slots=True)
class Invocation:
    """One request to run a builder.

    `resolved_configuration` is None when the run uses the builder's current
    defaults; a provider may resolve it from its own job's configuration instead.
    """

    inputs: Annotated[dict[str, BuilderInput], Field(strict=True)]
    resolved_configuration: _JsonObject | None
    input_access: InputAccessContext
    deadline_at: Deadline
    output_path: _NonEmptyString | None = None


@dataclass(frozen=True, slots=True)
class InlineOutput:
    value: JsonValue = None
    kind: Literal[OutputKind.INLINE] = OutputKind.INLINE


@dataclass(frozen=True, slots=True)
class SharedArtifact:
    """A completed output file; the result carries its path, not its bytes."""

    path: _NonEmptyString
    media_type: Annotated[str, Field(strict=True)] | None = None
    kind: Literal[OutputKind.SHARED_ARTIFACT] = OutputKind.SHARED_ARTIFACT


type BuilderOutput = Annotated[
    InlineOutput | SharedArtifact, Field(discriminator="kind")
]


@dataclass(frozen=True, slots=True)
class BuilderResult:
    outputs: Annotated[dict[str, BuilderOutput], Field(strict=True)]


INVOCATION: TypeAdapter[Invocation] = TypeAdapter(Invocation)
RESULT: TypeAdapter[BuilderResult] = TypeAdapter(BuilderResult)


@dataclass(frozen=True, slots=True)
class Artifact:
    """Declares a `SharedArtifact` output field: its media type and default file name.

    Used as `Annotated[SharedArtifact, Artifact(...)]` in a `BuilderOutputs` model.
    """

    media_type: str | None = None
    # Output file name when the invocation names no output path.
    default_path: str | None = None


class BuilderOutputs(BaseModel):
    """A builder's outputs, one field per output.

    A field is a `SharedArtifact`, annotated with `Artifact`, or an
    `InlineOutput`. A builder returns its own subclass; the registry derives the
    builder's output declaration from the fields, and the application turns it
    into the generic `BuilderResult` that is sent and stored.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    def result(self) -> BuilderResult:
        """Return the outputs as the generic result that is sent and stored.

        Returns:
            A `BuilderResult` with one entry per field.
        """
        return BuilderResult(outputs=dict(self))


@dataclass(frozen=True, slots=True)
class Input:
    """Declares a `BuilderInputs` field's media type.

    Used as `Annotated[Path, Input(...)]`, or `Annotated[Path | None, Input(...)]`
    for an optional input.
    """

    media_type: str | None = None


class BuilderInputs(BaseModel):
    """A builder's inputs, one field per input, as local files.

    A field is a `Path` (required) or `Path | None` defaulting to None
    (optional), annotated with `Input`. The registry derives the builder's input
    declaration from the fields; the application resolves each supplied input to
    a local file through the input access and hands the builder an instance.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    def paths(self) -> tuple[Path, ...]:
        """Return the supplied inputs' paths.

        Returns:
            One path per input the invocation supplied, in field order.
        """
        return tuple(path for path in dict(self).values() if path is not None)


class FailureCode(StrEnum):
    DEADLINE_EXCEEDED = "builder_deadline_exceeded"
    INTERNAL_ERROR = "builder_internal_error"
    INPUT_UNAVAILABLE = "input_unavailable"
    MODEL_UNAVAILABLE = "model_unavailable"
    MODEL_FAILED = "model_failed"
    MODEL_CONTRACT_MISMATCH = "model_contract_mismatch"
    INVALID_MODEL_OUTPUT = "invalid_model_output"


class InvalidInvocation(ValueError):  # noqa: N818
    """Inputs, access, configuration or deadline violate the builder's contract."""


class UnknownBuilderError(KeyError):
    pass


class UnknownModelError(KeyError):
    pass


class IncompatibleModelError(InvalidInvocation):
    """The model's inference contract is not the one the builder accepts."""


class ContractViolation(RuntimeError):  # noqa: N818
    """A builder returned outputs that differ from its declaration."""


class BuilderFailure(RuntimeError):  # noqa: N818
    """A caller-safe failure with a stable code and a retry decision."""

    def __init__(self, *, code: str, safe_message: str, retryable: bool) -> None:
        """Record a caller-safe failure.

        Args:
            code: Stable failure code, usually a `FailureCode`.
            safe_message: Message safe to show callers.
            retryable: Whether a retry may succeed.
        """
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message
        self.retryable = retryable
