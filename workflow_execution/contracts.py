from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Literal


type JsonValue = (
    bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None
)


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


def invocation_from_dict(value: object) -> WorkflowInvocation:
    mapping = _mapping(value, "workflow invocation")
    raw_inputs = _mapping(mapping.get("inputs"), "inputs")
    inputs = {name: _input(item) for name, item in raw_inputs.items()}
    configuration = _json_mapping(
        mapping.get("resolved_configuration"), "resolved_configuration"
    )
    raw_access = _mapping(mapping.get("input_access"), "input_access")
    access_kind = raw_access.get("kind")
    if not isinstance(access_kind, str) or not access_kind:
        raise ValueError("input_access.kind must be a non-empty string")
    input_access = InputAccessContext(
        kind=access_kind,
        state=_json_mapping(raw_access.get("state"), "input_access.state"),
    )
    raw_deadline = mapping.get("deadline_at")
    if not isinstance(raw_deadline, str):
        raise TypeError("deadline_at must be an ISO 8601 string")
    try:
        deadline_at = datetime.fromisoformat(raw_deadline)
    except ValueError as error:
        raise ValueError("deadline_at must be an ISO 8601 string") from error
    return WorkflowInvocation(
        inputs=inputs,
        resolved_configuration=configuration,
        input_access=input_access,
        deadline_at=deadline_at,
    )


def invocation_to_dict(invocation: WorkflowInvocation) -> dict[str, object]:
    return {
        "inputs": {name: asdict(value) for name, value in invocation.inputs.items()},
        "resolved_configuration": invocation.resolved_configuration,
        "input_access": asdict(invocation.input_access),
        "deadline_at": invocation.deadline_at.isoformat(),
    }


def result_to_dict(result: WorkflowResult) -> dict[str, object]:
    return {
        "outputs": {name: asdict(output) for name, output in result.outputs.items()}
    }


def result_from_dict(value: object) -> WorkflowResult:
    mapping = _mapping(value, "workflow result")
    raw_outputs = _mapping(mapping.get("outputs"), "outputs")
    return WorkflowResult(
        outputs={name: _output(output) for name, output in raw_outputs.items()}
    )


def is_json_value(value: object) -> bool:
    if value is None or isinstance(value, bool | int | float | str):
        return True
    if isinstance(value, list):
        return all(is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and is_json_value(item) for key, item in value.items()
        )
    return False


def _input(value: object) -> WorkflowInput:
    mapping = _mapping(value, "workflow input")
    kind = mapping.get("kind")
    if kind == "inline":
        inline_value = mapping.get("value")
        _ensure_json(inline_value, "inline input")
        return InlineInput(value=inline_value)
    if kind == "shared_path":
        path = mapping.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError("shared path input requires a non-empty path")
        return SharedPathInput(path=path)
    if kind == "provider":
        name = mapping.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("provider input requires a non-empty name")
        return ProviderInput(name=name)
    raise ValueError("workflow input has an unknown kind")


def _output(value: object) -> WorkflowOutput:
    mapping = _mapping(value, "workflow output")
    kind = mapping.get("kind")
    if kind == "inline":
        inline_value = mapping.get("value")
        _ensure_json(inline_value, "inline output")
        return InlineOutput(value=inline_value)
    if kind == "shared_artifact":
        path = mapping.get("path")
        media_type = mapping.get("media_type")
        if not isinstance(path, str) or not path:
            raise ValueError("shared artifact requires a non-empty path")
        if media_type is not None and not isinstance(media_type, str):
            raise ValueError("shared artifact media_type must be a string or null")
        return SharedArtifact(path=path, media_type=media_type)
    raise ValueError("workflow output has an unknown kind")


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object")
    return value


def _json_mapping(value: object, name: str) -> dict[str, JsonValue]:
    mapping = _mapping(value, name)
    _ensure_json(mapping, name)
    return mapping


def _ensure_json(value: object, name: str) -> None:
    if value is None or isinstance(value, bool | int | float | str):
        return
    if isinstance(value, list):
        for item in value:
            _ensure_json(item, name)
        return
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        for item in value.values():
            _ensure_json(item, name)
        return
    raise ValueError(f"{name} must contain only JSON-compatible values")
