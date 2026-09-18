from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime  # noqa: TC003 - FastAPI resolves dataclass fields.
from enum import StrEnum
from typing import Literal


type JsonValue = (
    bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None
)


class WorkflowFailureCode(StrEnum):
    DEADLINE_EXCEEDED = "workflow_deadline_exceeded"
    INTERNAL_ERROR = "workflow_internal_error"


@dataclass(frozen=True, slots=True)
class InlineInput:
    value: JsonValue
    kind: Literal["inline"] = "inline"


@dataclass(frozen=True, slots=True)
class SharedPathInput:
    path: str
    kind: Literal["shared_path"] = "shared_path"


@dataclass(frozen=True, slots=True)
class ProviderInput:
    name: str
    kind: Literal["provider"] = "provider"


type WorkflowInput = InlineInput | SharedPathInput | ProviderInput


@dataclass(frozen=True, slots=True)
class InputAccessContext:
    kind: str
    state: dict[str, JsonValue]


@dataclass(frozen=True, slots=True)
class WorkflowInvocation:
    inputs: dict[str, WorkflowInput]
    resolved_configuration: dict[str, JsonValue]
    input_access: InputAccessContext
    deadline_at: datetime


@dataclass(frozen=True, slots=True)
class InlineOutput:
    value: JsonValue
    kind: Literal["inline"] = "inline"


@dataclass(frozen=True, slots=True)
class SharedArtifact:
    path: str
    media_type: str | None = None
    kind: Literal["shared_artifact"] = "shared_artifact"


type WorkflowOutput = InlineOutput | SharedArtifact


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    outputs: dict[str, WorkflowOutput]


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
