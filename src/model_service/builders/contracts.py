"""Builder invocations, model bindings, results and safe errors.

Everything here crosses a process boundary (facade → scheduler database → Ray
builder replica), so the types are strict and JSON-serializable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

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
from pydantic.dataclasses import dataclass as validated_dataclass


_JSON_VALUE: TypeAdapter[PydanticJsonValue] = TypeAdapter(
    PydanticJsonValue, config=ConfigDict(allow_inf_nan=False)
)


def _validate_json_value(value: object) -> PydanticJsonValue:
    return _JSON_VALUE.validate_python(value, strict=True)


type JsonValue = Annotated[PydanticJsonValue, BeforeValidator(_validate_json_value)]
type _NonEmptyString = Annotated[str, Field(strict=True, min_length=1)]
type _JsonObject = Annotated[dict[str, JsonValue], Field(strict=True)]


def is_json_value(value: object) -> bool:
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
    model_config = ConfigDict(frozen=True, extra="forbid")

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
    """HTTP model returning one float32 vector per summary name for an RGB tile."""

    contract: Literal[InferenceContract.TILE_EMBEDDING] = (
        InferenceContract.TILE_EMBEDDING
    )
    summaries: Annotated[
        tuple[Annotated[str, Field(min_length=1)], ...], Field(min_length=1)
    ]


type ModelBinding = Annotated[
    SpatialScoresModel | TileEmbeddingModel, Field(discriminator="contract")
]
MODEL_BINDING: TypeAdapter[ModelBinding] = TypeAdapter(ModelBinding)


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
    """An input the provider's access adapter materializes, e.g. an EMPAIA input."""

    name: _NonEmptyString
    kind: Literal[InputKind.PROVIDER] = InputKind.PROVIDER


type BuilderInput = Annotated[
    SharedPathInput | ProviderInput, Field(discriminator="kind")
]


@dataclass(frozen=True, slots=True)
class InputAccessContext:
    """Selects an access adapter (`shared_storage`, `empaia`) and its public state."""

    kind: _NonEmptyString
    state: _JsonObject


@validated_dataclass(frozen=True, slots=True, config=ConfigDict(extra="forbid"))
class ExecutionContext:
    """Attempt identity and secret provider state, kept out of the stored invocation."""

    processing_attempt_id: _NonEmptyString | None = None
    protected_context: _JsonObject = field(default_factory=dict, repr=False)


def _iso_datetime(value: object) -> datetime:
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
    """One request to run a builder; stored durably for scheduled executions.

    `resolved_configuration` is None for standalone work, which uses the builder's
    current defaults when each attempt starts. EMPAIA admission resolves it from
    the job's configuration.
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


class FailureCode(StrEnum):
    DEADLINE_EXCEEDED = "builder_deadline_exceeded"
    INTERNAL_ERROR = "builder_internal_error"
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
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message
        self.retryable = retryable
