from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    TypeAdapter,
    ValidationError,
)
from pydantic import JsonValue as PydanticJsonValue


_JSON_VALUE: TypeAdapter[PydanticJsonValue] = TypeAdapter(
    PydanticJsonValue, config=ConfigDict(allow_inf_nan=False)
)


def _validate_json_value(value: object) -> PydanticJsonValue:
    return _JSON_VALUE.validate_python(value, strict=True)


type JsonValue = Annotated[PydanticJsonValue, BeforeValidator(_validate_json_value)]
type _NonEmptyString = Annotated[str, Field(strict=True, min_length=1)]
type _JsonObject = Annotated[dict[str, JsonValue], Field(strict=True)]


def _iso_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        # Pydantic wraps ValueError as validation failure; TypeError escapes it.
        raise ValueError("deadline_at must be an ISO 8601 string")  # noqa: TRY004
    return datetime.fromisoformat(value)


class WorkflowFailureCode(StrEnum):
    DEADLINE_EXCEEDED = "workflow_deadline_exceeded"
    INTERNAL_ERROR = "workflow_internal_error"


class WorkflowInputKind(StrEnum):
    INLINE = "inline"
    SHARED_PATH = "shared_path"
    PROVIDER = "provider"


class WorkflowOutputKind(StrEnum):
    INLINE = "inline"
    SHARED_ARTIFACT = "shared_artifact"


@dataclass(frozen=True, slots=True)
class InlineInput:
    value: JsonValue = None
    kind: Literal[WorkflowInputKind.INLINE] = WorkflowInputKind.INLINE


@dataclass(frozen=True, slots=True)
class SharedPathInput:
    path: _NonEmptyString
    kind: Literal[WorkflowInputKind.SHARED_PATH] = WorkflowInputKind.SHARED_PATH


@dataclass(frozen=True, slots=True)
class ProviderInput:
    name: _NonEmptyString
    kind: Literal[WorkflowInputKind.PROVIDER] = WorkflowInputKind.PROVIDER


type WorkflowInput = Annotated[
    InlineInput | SharedPathInput | ProviderInput, Field(discriminator="kind")
]


@dataclass(frozen=True, slots=True)
class InputAccessContext:
    kind: _NonEmptyString
    state: _JsonObject


@dataclass(frozen=True, slots=True)
class WorkflowInvocation:
    inputs: Annotated[dict[str, WorkflowInput], Field(strict=True)]
    resolved_configuration: _JsonObject
    input_access: InputAccessContext
    deadline_at: Annotated[
        datetime,
        BeforeValidator(_iso_datetime),
        PlainSerializer(datetime.isoformat, return_type=str),
    ]


@dataclass(frozen=True, slots=True)
class InlineOutput:
    value: JsonValue = None
    kind: Literal[WorkflowOutputKind.INLINE] = WorkflowOutputKind.INLINE


@dataclass(frozen=True, slots=True)
class SharedArtifact:
    path: _NonEmptyString
    media_type: Annotated[str, Field(strict=True)] | None = None
    kind: Literal[WorkflowOutputKind.SHARED_ARTIFACT] = (
        WorkflowOutputKind.SHARED_ARTIFACT
    )


type WorkflowOutput = Annotated[
    InlineOutput | SharedArtifact, Field(discriminator="kind")
]


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    outputs: Annotated[dict[str, WorkflowOutput], Field(strict=True)]


def is_json_value(value: object) -> bool:
    try:
        _JSON_VALUE.validate_python(value, strict=True)
    except ValidationError:
        return False
    return True


class InvalidWorkflowInvocation(ValueError):  # noqa: N818 - Contract name.
    pass


class WorkflowContractViolation(RuntimeError):  # noqa: N818 - Contract name.
    pass


class WorkflowFailure(RuntimeError):  # noqa: N818 - Contract name.
    def __init__(self, *, code: str, safe_message: str, retryable: bool) -> None:
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message
        self.retryable = retryable
