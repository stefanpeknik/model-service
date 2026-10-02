from __future__ import annotations

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from datetime import datetime


type JsonValue = (
    None | bool | int | float | str | list[JsonValue] | dict[str, JsonValue]
)
_SENSITIVE_NAMES = {
    "authorization",
    "credential",
    "credentials",
    "password",
    "secret",
    "token",
}
_SENSITIVE_NAME_PATTERN = re.compile(
    r"(?:^|[_-])(?:authorization|credentials?|password|secret|token)(?:$|[_-])",
    re.IGNORECASE,
)


class ExecutionState(StrEnum):
    """Durable lifecycle states exposed by the control-plane API."""

    ACCEPTED = "ACCEPTED"
    RESOLVING = "RESOLVING"
    READY = "READY"
    RUNNING = "RUNNING"
    OUTPUT_READY = "OUTPUT_READY"
    PUBLISHING = "PUBLISHING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMED_OUT = "TIMED_OUT"

    @property
    def terminal(self) -> bool:
        return self in {
            ExecutionState.COMPLETED,
            ExecutionState.FAILED,
            ExecutionState.CANCELLED,
            ExecutionState.TIMED_OUT,
        }


class FailureCategory(StrEnum):
    APPLICATION = "APPLICATION"
    INFRASTRUCTURE = "INFRASTRUCTURE"


class TimeoutKind(StrEnum):
    INITIAL_CAPACITY = "INITIAL_CAPACITY"
    MODEL_RUNTIME = "MODEL_RUNTIME"


@dataclass(frozen=True, slots=True)
class AppMapping:
    """Deployment-controlled snapshot selected during admission."""

    app_id: str
    app_version: str
    model_id: str
    model_version: str
    workflow: str
    resource_profile: str
    max_runtime_seconds: int
    max_attempts: int = 3
    input_name: str = "slide"
    output_name: str = "heatmap"
    model_parameter_names: tuple[str, ...] = ()
    allowed_wsi_plugin_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "app_id",
            "app_version",
            "model_id",
            "model_version",
            "workflow",
            "resource_profile",
            "input_name",
            "output_name",
        ):
            _require_identifier(name, getattr(self, name))
        for parameter_name in self.model_parameter_names:
            _require_identifier("model_parameter_name", parameter_name)
            if _is_sensitive_name(parameter_name):
                raise ValueError("credentials cannot be configured as model parameters")
        for plugin_id in self.allowed_wsi_plugin_ids:
            _require_identifier("allowed_wsi_plugin_id", plugin_id)
        if len(set(self.model_parameter_names)) != len(self.model_parameter_names):
            raise ValueError("model parameter names must be unique")
        if len(set(self.allowed_wsi_plugin_ids)) != len(self.allowed_wsi_plugin_ids):
            raise ValueError("allowed WSI plugin IDs must be unique")
        if (
            self.max_runtime_seconds <= 0
            or self.max_attempts <= 0
            or self.max_attempts > 100
        ):
            raise ValueError(
                "runtime must be positive and max attempts must be between 1 and 100"
            )


@dataclass(frozen=True, slots=True)
class AdmissionRequest:
    installation_alias: str
    empaia_job_id: str
    registered_app_id: str
    registered_app_version: str
    idempotency_key: str

    def __post_init__(self) -> None:
        for name in (
            "installation_alias",
            "empaia_job_id",
            "registered_app_id",
            "registered_app_version",
            "idempotency_key",
        ):
            _require_identifier(name, getattr(self, name))


@dataclass(frozen=True, slots=True)
class ExecutionSpec:
    """Versioned, credential-free specification consumed by a WSI worker."""

    model_id: str
    model_version: str
    workflow: str
    resource_profile: str
    max_runtime_seconds: int
    wsi_id: str
    storage_id: str
    relative_locator: str
    output_name: str
    model_parameters: dict[str, JsonValue]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("unsupported execution specification version")
        if self.max_runtime_seconds <= 0:
            raise ValueError("execution runtime limit must be positive")
        for name in (
            "model_id",
            "model_version",
            "workflow",
            "resource_profile",
            "wsi_id",
            "storage_id",
            "output_name",
        ):
            _require_identifier(name, getattr(self, name))
        _require_relative_path("relative_locator", self.relative_locator)
        _reject_sensitive_keys(self.model_parameters)
        json.dumps(self.model_parameters, allow_nan=False)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "workflow": self.workflow,
            "resource_profile": self.resource_profile,
            "max_runtime_seconds": self.max_runtime_seconds,
            "wsi_id": self.wsi_id,
            "storage_id": self.storage_id,
            "relative_locator": self.relative_locator,
            "output_name": self.output_name,
            "model_parameters": self.model_parameters,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ExecutionSpec:
        expected = {
            "schema_version",
            "model_id",
            "model_version",
            "workflow",
            "resource_profile",
            "max_runtime_seconds",
            "wsi_id",
            "storage_id",
            "relative_locator",
            "output_name",
            "model_parameters",
        }
        if set(value) != expected or not isinstance(value["model_parameters"], dict):
            raise ValueError("invalid execution specification")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class Execution:
    id: str
    installation_alias: str
    empaia_job_id: str
    registered_app_id: str
    registered_app_version: str
    idempotency_key: str
    request_fingerprint: str
    model_id: str
    model_version: str
    workflow: str
    resource_profile: str
    max_runtime_seconds: int
    max_attempts: int
    input_name: str
    output_name: str
    model_parameter_names: tuple[str, ...]
    allowed_wsi_plugin_ids: tuple[str, ...]
    state: ExecutionState
    created_at: datetime
    updated_at: datetime
    resolution_after: datetime | None
    ready_at: datetime | None
    first_claimed_at: datetime | None
    current_attempt_id: str | None
    attempt_heartbeat_at: datetime | None
    dispatch_after: datetime | None
    attempt_count: int
    cancellation_requested: bool
    failure_category: FailureCategory | None
    timeout_kind: TimeoutKind | None
    safe_reason_code: str | None
    safe_reason_message: str | None
    execution_spec: ExecutionSpec | None


@dataclass(frozen=True, slots=True)
class AdmissionResult:
    execution: Execution
    created: bool


class ExecutionStoreError(RuntimeError):
    """Base error for durable lifecycle operations."""


class ExecutionNotFoundError(ExecutionStoreError):
    pass


class SubmissionConflictError(ExecutionStoreError):
    pass


class InvalidTransitionError(ExecutionStoreError):
    pass


class AttemptOwnershipError(ExecutionStoreError):
    pass


class IncompatibleSchemaError(ExecutionStoreError):
    pass


def _require_identifier(name: str, value: str) -> None:
    if (
        not value
        or len(value) > 255
        or value != value.strip()
        or any(character.isspace() for character in value)
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{name} must be a short identifier without whitespace")


def _require_relative_path(name: str, value: str) -> None:
    if not value or "\\" in value or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty POSIX relative path")
    raw_parts = value.split("/")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError(f"{name} must not be absolute or contain traversal")


def _reject_sensitive_keys(value: JsonValue) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if _is_sensitive_name(key):
                raise ValueError("credentials cannot be stored in model parameters")
            _reject_sensitive_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_sensitive_keys(child)


def _is_sensitive_name(value: str) -> bool:
    return value.casefold() in _SENSITIVE_NAMES or bool(
        _SENSITIVE_NAME_PATTERN.search(value)
    )
