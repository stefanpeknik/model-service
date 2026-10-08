"""Define prototype admissions, pinned execution specifications, and its separate lifecycle vocabulary.

Historical SQLite control-plane prototype. The current production scheduler lives in
execution_scheduler/ and uses PostgreSQL; these lifecycle rules are not its
specification.
"""

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
        """Identify completed, failed, cancelled, and timed-out states that cannot resume active work.

        Returns:
            bool: True when the described check succeeds; otherwise False.
        """
        return self in {
            ExecutionState.COMPLETED,
            ExecutionState.FAILED,
            ExecutionState.CANCELLED,
            ExecutionState.TIMED_OUT,
        }


class FailureCategory(StrEnum):
    """Distinguish unsupported job data from control-plane infrastructure failures."""

    APPLICATION = "APPLICATION"
    INFRASTRUCTURE = "INFRASTRUCTURE"


class TimeoutKind(StrEnum):
    """Distinguish initial capacity waiting from runtime measured since the first claim."""

    INITIAL_CAPACITY = "INITIAL_CAPACITY"
    MODEL_RUNTIME = "MODEL_RUNTIME"


@dataclass(frozen=True, slots=True)
class AppMapping:
    """Deployment-controlled snapshot selected during admission.

    Attributes:
        app_id: Registered EMPAIA app identifier.
        app_version: Registered app version permitted by the mapping.
        model_id: Pinned classifier application identifier.
        model_version: Pinned model version from prototype admission.
        workflow: Pinned prototype computation workflow identifier.
        resource_profile: Pinned prototype compute resource class.
        max_runtime_seconds: Prototype runtime budget measured from the first claim.
        max_attempts: Maximum consumed attempts permitted by this policy.
        input_name: Named prototype external slide input.
        output_name: Named prototype result output.
        model_parameter_names: Allow-listed model configuration names selected during
            resolution.
        allowed_wsi_plugin_ids: WSI representations enabled by the prototype app
            mapping.
    """

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
        """Validate mapping identifiers, permitted parameter/plugin names, positive runtime, and a bounded attempt limit.

        Raises:
            ValueError: A value fails the validation described above.
        """
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
    """Carry immutable installation/job/app identity and the caller replay key.

    Attributes:
        installation_alias: Deployment-approved prototype installation identity.
        empaia_job_id: External job identity pinned by prototype admission.
        registered_app_id: External registered application identity in the prototype
            request.
        registered_app_version: Exact external registered application version.
        idempotency_key: Caller replay key; required for standalone admission.
    """

    installation_alias: str
    empaia_job_id: str
    registered_app_id: str
    registered_app_version: str
    idempotency_key: str

    def __post_init__(self) -> None:
        """Require short nonempty identity values without whitespace or control characters."""
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
    """Versioned, credential-free specification consumed by a WSI worker.

    Attributes:
        model_id: Pinned classifier application identifier.
        model_version: Pinned model version from prototype admission.
        workflow: Pinned prototype computation workflow identifier.
        resource_profile: Pinned prototype compute resource class.
        max_runtime_seconds: Prototype runtime budget measured from the first claim.
        wsi_id: Resolved provider slide identity.
        storage_id: Configured worker storage mapping identifier.
        relative_locator: Traversal-free file locator beneath the authorized worker
            mount.
        output_name: Named prototype result output.
        model_parameters: Finite credential-free allow-listed model settings.
        schema_version: Version of the pinned prototype worker specification.
    """

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
        """Require the supported schema version, pinned identifiers/runtime, a relative locator, and finite credential-free model parameters.

        Raises:
            ValueError: A value fails the validation described above.
        """
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
        """Serialize the complete pinned worker specification as JSON-compatible fields.

        Returns:
            dict[str, JsonValue]: Decoded, serialized, or recorded fields described
                above.
        """
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
        """Require exactly the supported specification fields and reconstruct the validated value.

        Args:
            value: Value being validated, serialized, converted, or checked by this
                helper.

        Returns:
            ExecutionSpec: Validated credential-free prototype worker specification.

        Raises:
            ValueError: A value fails the validation described above.
        """
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
    """Snapshot prototype lifecycle state, its pinned app contract, resolution data, and worker ownership.

    Attributes:
        id: Execution or provider object identifier, according to this record.
        installation_alias: Deployment-approved prototype installation identity.
        empaia_job_id: External job identity pinned by prototype admission.
        registered_app_id: External registered application identity in the prototype
            request.
        registered_app_version: Exact external registered application version.
        idempotency_key: Caller replay key; required for standalone admission.
        request_fingerprint: Canonical digest distinguishing identical replay contents.
        model_id: Pinned classifier application identifier.
        model_version: Pinned model version from prototype admission.
        workflow: Pinned prototype computation workflow identifier.
        resource_profile: Pinned prototype compute resource class.
        max_runtime_seconds: Prototype runtime budget measured from the first claim.
        max_attempts: Maximum consumed attempts permitted by this policy.
        input_name: Named prototype external slide input.
        output_name: Named prototype result output.
        model_parameter_names: Allow-listed model configuration names selected during
            resolution.
        allowed_wsi_plugin_ids: WSI representations enabled by the prototype app
            mapping.
        state: Public access-adapter JSON or historical prototype lifecycle state.
        created_at: Initial durable admission timestamp.
        updated_at: Timestamp of the latest durable state change.
        resolution_after: Earliest next prototype provider-resolution time.
        ready_at: Timestamp the prototype specification became dispatchable.
        first_claimed_at: Original compute claim time, preserved across retries for
            runtime enforcement.
        current_attempt_id: Prototype worker identity currently owning computation.
        attempt_heartbeat_at: Last prototype worker ownership heartbeat.
        dispatch_after: Earliest retry dispatch time in the prototype.
        attempt_count: Number of prototype compute claims consumed.
        cancellation_requested: Whether prototype cancellation intent has been recorded.
        failure_category: Prototype application/infrastructure failure classification.
        timeout_kind: Prototype initial-capacity or runtime expiry category.
        safe_reason_code: Bounded prototype terminal reason identifier.
        safe_reason_message: Bounded safe prototype terminal reason explanation.
        execution_spec: Immutable resolved prototype worker inputs, if ready.
    """

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
    """Return a durable execution with insertion-versus-replay information.

    Attributes:
        execution: Internal or public execution snapshot, according to this decision.
        created: True for insertion and False for an identical replay.
    """

    execution: Execution
    created: bool


class ExecutionStoreError(RuntimeError):
    """Base error for durable lifecycle operations."""


class ExecutionNotFoundError(ExecutionStoreError):
    """Signal a prototype operation on an absent execution ID."""


class SubmissionConflictError(ExecutionStoreError):
    """Signal a replay key or external job identity reused with conflicting admission data."""


class InvalidTransitionError(ExecutionStoreError):
    """Reject a state change that contradicts the prototype lifecycle or pinned specification."""


class AttemptOwnershipError(ExecutionStoreError):
    """Reject a worker action that no longer owns the current attempt."""


class IncompatibleSchemaError(ExecutionStoreError):
    """Signal an unsupported SQLite schema version or malformed stored data."""


def _require_identifier(name: str, value: str) -> None:
    """Reject empty, oversized, whitespace-containing, or control-character identity values.

    Args:
        name: Input/output name or configuration field selected by this operation.
        value: Value being validated, serialized, converted, or checked by this helper.

    Returns:
        None: No return value.

    Raises:
        ValueError: A value fails the validation described above.
    """
    if (
        not value
        or len(value) > 255
        or value != value.strip()
        or any(character.isspace() for character in value)
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{name} must be a short identifier without whitespace")


def _require_relative_path(name: str, value: str) -> None:
    """Require a nonempty traversal-free POSIX relative locator.

    Args:
        name: Input/output name or configuration field selected by this operation.
        value: Value being validated, serialized, converted, or checked by this helper.

    Returns:
        None: No return value.

    Raises:
        ValueError: A value fails the validation described above.
    """
    if not value or "\\" in value or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty POSIX relative path")
    raw_parts = value.split("/")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in raw_parts):
        raise ValueError(f"{name} must not be absolute or contain traversal")


def _reject_sensitive_keys(value: JsonValue) -> None:
    """Recursively reject credential-like keys in stored model parameters.

    Args:
        value: Value being validated, serialized, converted, or checked by this helper.

    Returns:
        None: No return value.

    Raises:
        ValueError: A value fails the validation described above.
    """
    if isinstance(value, dict):
        for key, child in value.items():
            if _is_sensitive_name(key):
                raise ValueError("credentials cannot be stored in model parameters")
            _reject_sensitive_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_sensitive_keys(child)


def _is_sensitive_name(value: str) -> bool:
    """Match credential-like names case-insensitively, including underscore/dash-separated name components.

    Args:
        value: Value being validated, serialized, converted, or checked by this helper.

    Returns:
        bool: True when the parameter name matches a credential-like name or component.
    """
    return value.casefold() in _SENSITIVE_NAMES or bool(
        _SENSITIVE_NAME_PATTERN.search(value)
    )
