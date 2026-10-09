"""Define provider-neutral workflow inputs, invocations, results, and safe errors.

Serializable invocation data is separate from protected execution context. Inputs
describe where data lives; results contain inline JSON or references to completed
artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
from pydantic.dataclasses import dataclass as validated_dataclass


_JSON_VALUE: TypeAdapter[PydanticJsonValue] = TypeAdapter(
    PydanticJsonValue, config=ConfigDict(allow_inf_nan=False)
)


def _validate_json_value(value: object) -> PydanticJsonValue:
    """Validate strict finite JSON recursively, rejecting unsupported Python objects and non-finite numbers.

    Args:
        value: Potential JSON value to validate recursively without coercion or non-
            finite numbers.

    Returns:
        PydanticJsonValue: Strictly validated finite JSON-compatible data.

    Raises:
        ValidationError: value contains unsupported Python objects or non-finite JSON
            numbers.
    """
    return _JSON_VALUE.validate_python(value, strict=True)


type JsonValue = Annotated[PydanticJsonValue, BeforeValidator(_validate_json_value)]
type _NonEmptyString = Annotated[str, Field(strict=True, min_length=1)]
type _JsonObject = Annotated[dict[str, JsonValue], Field(strict=True)]


def _iso_datetime(value: object) -> datetime:
    """Accept a datetime or parse an ISO 8601 string; reject other types as Pydantic validation errors.

    Args:
        value: An existing datetime or ISO 8601 string; other types are rejected.

    Returns:
        datetime: Existing datetime or parsed ISO 8601 timestamp; timezone checks occur
            during invocation validation.

    Raises:
        ValueError: value is neither a datetime nor a valid ISO 8601 timestamp string.
    """
    if isinstance(value, datetime):
        return value
    if not isinstance(value, str):
        # Pydantic wraps ValueError as validation failure; TypeError escapes it.
        raise ValueError("deadline_at must be an ISO 8601 string")
    return datetime.fromisoformat(value)


class WorkflowFailureCode(StrEnum):
    """Name runtime failures for deadlines, unavailable settings, and unexpected errors."""

    DEADLINE_EXCEEDED = "workflow_deadline_exceeded"
    INTERNAL_ERROR = "workflow_internal_error"
    SETTINGS_UNAVAILABLE = "workflow_settings_unavailable"


class WorkflowInputKind(StrEnum):
    """Distinguish inline values, shared filesystem paths, and provider-resolved input names."""

    INLINE = "inline"
    SHARED_PATH = "shared_path"
    PROVIDER = "provider"


class WorkflowOutputKind(StrEnum):
    """Distinguish inline JSON outputs from references to shared artifact files."""

    INLINE = "inline"
    SHARED_ARTIFACT = "shared_artifact"


@dataclass(frozen=True, slots=True)
class InlineInput:
    """Carry a JSON value directly in the invocation rather than referencing external data.

    Attributes:
        value: Finite JSON value carried directly in the invocation.
        kind: Discriminator fixed to inline.
    """

    value: JsonValue = None
    kind: Literal[WorkflowInputKind.INLINE] = WorkflowInputKind.INLINE


@dataclass(frozen=True, slots=True)
class SharedPathInput:
    """Reference an existing input file visible through shared storage.

    Attributes:
        path: Input file path visible through the invocation's shared storage root.
        kind: Discriminator fixed to shared_path.
    """

    path: _NonEmptyString
    kind: Literal[WorkflowInputKind.SHARED_PATH] = WorkflowInputKind.SHARED_PATH


@dataclass(frozen=True, slots=True)
class ProviderInput:
    """Name an input that a provider-specific access adapter must resolve or materialize.

    Attributes:
        name: External provider input name for the access adapter to resolve.
        kind: Discriminator fixed to provider.
    """

    name: _NonEmptyString
    kind: Literal[WorkflowInputKind.PROVIDER] = WorkflowInputKind.PROVIDER


type WorkflowInput = Annotated[
    InlineInput | SharedPathInput | ProviderInput, Field(discriminator="kind")
]


@dataclass(frozen=True, slots=True)
class InputAccessContext:
    """Select an access adapter and carry its serializable, non-secret configuration.

    Attributes:
        kind: Registered adapter identifier, such as shared_storage or empaia.
        state: Public JSON settings consumed by that adapter, excluding protected
            credentials.
    """

    kind: _NonEmptyString
    state: _JsonObject


@validated_dataclass(frozen=True, slots=True, config=ConfigDict(extra="forbid"))
class WorkflowExecutionContext:
    """Carry attempt identity and private provider state separately from the public invocation. Protected context is hidden from repr.

    Attributes:
        processing_attempt_id: Optional private claim token used for output isolation.
        protected_context: Private provider JSON, encrypted in PostgreSQL and excluded
            from public views.
    """

    processing_attempt_id: _NonEmptyString | None = None
    protected_context: _JsonObject = field(default_factory=dict, repr=False)


type WorkflowDeadline = Annotated[
    datetime,
    BeforeValidator(_iso_datetime),
    PlainSerializer(datetime.isoformat, return_type=str),
]


@dataclass(frozen=True, slots=True)
class WorkflowInvocation:
    """Describe named inputs, resolved configuration, input access, and an absolute execution deadline.

    Attributes:
        inputs: Input names mapped to the invocation's typed input values.
        resolved_configuration: Complete provider-prepared settings, or None for
            standalone work that uses server defaults when processing starts.
        input_access: Public access adapter selection and non-secret state.
        deadline_at: Absolute aware deadline, including input acquisition and execution.
        output_path: Optional standalone destination override, not processing settings.
    """

    inputs: Annotated[dict[str, WorkflowInput], Field(strict=True)]
    resolved_configuration: _JsonObject | None
    input_access: InputAccessContext
    deadline_at: WorkflowDeadline
    output_path: _NonEmptyString | None = None


@dataclass(frozen=True, slots=True)
class InlineOutput:
    """Return a JSON-compatible value directly in the workflow result.

    Attributes:
        value: Finite JSON value carried directly in the result.
        kind: Discriminator fixed to inline.
    """

    value: JsonValue = None
    kind: Literal[WorkflowOutputKind.INLINE] = WorkflowOutputKind.INLINE


@dataclass(frozen=True, slots=True)
class SharedArtifact:
    """Reference a completed output file and optional media type; the result does not carry file bytes.

    Attributes:
        path: Completed output file path accessible through the invocation's storage
            boundary.
        media_type: Optional output content type, such as image/tiff.
        kind: Discriminator fixed to shared_artifact.
    """

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
    """Map declared output names to inline values or shared artifact references.

    Attributes:
        outputs: Output names mapped to returned inline values or shared artifacts.
    """

    outputs: Annotated[dict[str, WorkflowOutput], Field(strict=True)]


def is_json_value(value: object) -> bool:
    """Check strict finite JSON compatibility without propagating validation errors.

    Args:
        value: Potential inline JSON value to check without propagating validation
            errors.

    Returns:
        bool: True for strict finite JSON-compatible values; False for unsupported
            values.
    """
    try:
        _JSON_VALUE.validate_python(value, strict=True)
    except ValidationError:
        return False
    return True


class InvalidWorkflowInvocation(ValueError):  # noqa: N818 - Contract name.
    """Signal invalid inputs, access settings, or deadlines before useful workflow execution."""


class WorkflowContractViolation(RuntimeError):  # noqa: N818 - Contract name.
    """Signal an implementation that returned outputs inconsistent with its declared workflow definition."""


class WorkflowFailure(RuntimeError):  # noqa: N818 - Contract name.
    """Carry a caller-safe workflow error code, message, and retry decision."""

    def __init__(self, *, code: str, safe_message: str, retryable: bool) -> None:
        """Store safe error metadata while retaining the original cause through normal exception chaining.

        Args:
            code: Stable failure identifier, or HTTP status when classifying an upstream
                response.
            safe_message: Message permitted in public errors without internal details or
                credentials.
            retryable: Whether another attempt is permitted for this classified failure.
        """
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message
        self.retryable = retryable
