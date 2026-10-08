"""Parse deployment-owned mappings, credential-file paths, and timing for the prototype.

Historical SQLite control-plane prototype. The current production scheduler lives in
execution_scheduler/ and uses PostgreSQL; these lifecycle rules are not its
specification.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from control_plane.domain import AppMapping
from control_plane.storage import StorageMapping


@dataclass(frozen=True, slots=True)
class Settings:
    """Hold SQLite, installation, credential, storage, app mapping, and reconciliation configuration.

    Attributes:
        database_path: Historical prototype SQLite file location.
        installation_alias: Deployment-approved prototype installation identity.
        jes_bearer_token_file: Prototype JES authentication credential file.
        app_mappings: Enabled prototype app/version contracts.
        app_service_url: Prototype AppService HTTP base origin.
        app_service_token_file: Separate prototype AppService credential file.
        storage_mappings: Configured prototype provider-prefix to worker-mount
            translations.
        allow_broad_app_service_credential: Explicit prototype opt-in to its broad
            AppService credential boundary.
        allow_insecure_app_service_http: Explicit prototype opt-in to plaintext provider
            transport.
        resolution_retry_seconds: Delay after temporary prototype resolution failures.
        max_initial_capacity_wait_seconds: Optional prototype budget before the first
            compute claim.
        reconcile_interval_seconds: Seconds between prototype reconciliation/resolution
            passes.
        worker_stale_after_seconds: Heartbeat age permitting prototype attempt recovery.
        retry_backoff_seconds: Delay before recovered prototype work is dispatchable.
    """

    database_path: Path
    installation_alias: str
    jes_bearer_token_file: Path
    app_mappings: tuple[AppMapping, ...]
    app_service_url: str
    app_service_token_file: Path
    storage_mappings: tuple[StorageMapping, ...]
    allow_broad_app_service_credential: bool = False
    allow_insecure_app_service_http: bool = False
    resolution_retry_seconds: float = 5.0
    max_initial_capacity_wait_seconds: int | None = None
    reconcile_interval_seconds: float = 5.0
    worker_stale_after_seconds: float = 60.0
    retry_backoff_seconds: float = 5.0

    @classmethod
    def from_env(cls) -> Settings:
        """Parse required prototype environment variables and optional timing/credential-boundary opt-ins.

        Returns:
            Settings: Parsed historical prototype deployment configuration.
        """
        mappings = _parse_app_mappings(_required_env("EMPAIA_APP_MAPPINGS_JSON"))
        return cls(
            database_path=Path(
                os.environ.get(
                    "EMPAIA_EXECUTION_DATABASE_PATH", "/var/lib/model-service/jobs.db"
                )
            ),
            installation_alias=_required_env("EMPAIA_INSTALLATION_ALIAS"),
            jes_bearer_token_file=_token_file("EMPAIA_JES_BEARER_TOKEN_FILE"),
            app_mappings=mappings,
            app_service_url=_required_env("EMPAIA_APP_SERVICE_URL"),
            app_service_token_file=_token_file("EMPAIA_APP_SERVICE_BEARER_TOKEN_FILE"),
            storage_mappings=_parse_storage_mappings(
                _required_env("EMPAIA_STORAGE_MAPPINGS_JSON")
            ),
            allow_broad_app_service_credential=_boolean(
                "EMPAIA_POC_ALLOW_BROAD_APP_SERVICE_CREDENTIAL", False
            ),
            allow_insecure_app_service_http=_boolean(
                "EMPAIA_POC_ALLOW_INSECURE_APP_SERVICE_HTTP", False
            ),
            resolution_retry_seconds=_positive_float(
                "EMPAIA_RESOLUTION_RETRY_SECONDS", 5.0
            ),
            max_initial_capacity_wait_seconds=_optional_positive_int(
                "EMPAIA_MAX_INITIAL_CAPACITY_WAIT_SECONDS"
            ),
            reconcile_interval_seconds=_positive_float(
                "EMPAIA_RECONCILE_INTERVAL_SECONDS", 5.0
            ),
            worker_stale_after_seconds=_positive_float(
                "EMPAIA_WORKER_STALE_AFTER_SECONDS", 60.0
            ),
            retry_backoff_seconds=_nonnegative_float(
                "EMPAIA_RETRY_BACKOFF_SECONDS", 5.0
            ),
        )


def _required_env(name: str) -> str:
    """Require a nonempty stripped environment value.

    Args:
        name: Input/output name or configuration field selected by this operation.

    Returns:
        str: Nonempty stripped required prototype environment setting.

    Raises:
        ValueError: A value fails the validation described above.
    """
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} must be configured")
    return value


def _token_file(variable_name: str) -> Path:
    """Return a configured credential path only after confirming it contains a nonempty token.

    Args:
        variable_name: Environment variable containing a prototype credential-file path.

    Returns:
        Path: Configured credential file path after checking the file contains a token.

    Raises:
        ValueError: A value fails the validation described above.
    """
    path = Path(_required_env(variable_name))
    token = path.read_text(encoding="utf-8").strip()
    if not token:
        raise ValueError(f"{variable_name} points to an empty file")
    return path


def _boolean(name: str, default: bool) -> bool:
    """Parse true/false or 1/0 case-insensitively, using the default when unset or blank.

    Args:
        name: Input/output name or configuration field selected by this operation.
        default: Fallback value used when the setting or optional field is absent.

    Returns:
        bool: True when the described check succeeds; otherwise False.

    Raises:
        ValueError: A value fails the validation described above.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    normalized = raw.strip().casefold()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise ValueError(f"{name} must be true or false")


def _optional_positive_int(name: str) -> int | None:
    """Parse a positive integer policy or return None when no limit is configured.

    Args:
        name: Input/output name or configuration field selected by this operation.

    Returns:
        int | None: Validated numeric setting or deterministic prediction described
            above. None indicates an absent, unsupported, or unmatched value as
            described above.

    Raises:
        ValueError: The configured value is not an integer greater than zero.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(name: str, default: float) -> float:
    """Parse a finite positive duration using the shared nonnegative parser.

    Args:
        name: Input/output name or configuration field selected by this operation.
        default: Fallback value used when the setting or optional field is absent.

    Returns:
        float: Validated numeric setting or deterministic prediction described above.

    Raises:
        ValueError: A value fails the validation described above.
    """
    value = _nonnegative_float(name, default)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_float(name: str, default: float) -> float:
    """Parse a finite nonnegative duration or its default.

    Args:
        name: Input/output name or configuration field selected by this operation.
        default: Fallback value used when the setting or optional field is absent.

    Returns:
        float: Validated numeric setting or deterministic prediction described above.

    Raises:
        ValueError: The configured value is not numeric, is non-finite, or is negative.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a non-negative number") from error
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative number")
    return value


def _parse_app_mappings(raw: str) -> tuple[AppMapping, ...]:
    """Validate a nonempty JSON list of unique app/version contracts and construct pinned app mappings.

    Args:
        raw: JSON text or environment value to parse.

    Returns:
        tuple[AppMapping, ...]: Validated unique app/version mappings in configuration
            order.

    Raises:
        ValueError: A value fails the validation described above.
    """
    parsed: Any = json.loads(raw)
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("EMPAIA_APP_MAPPINGS_JSON must be a non-empty JSON list")

    mappings: list[AppMapping] = []
    keys: set[tuple[str, str]] = set()
    required_fields = {
        "app_id",
        "app_version",
        "model_id",
        "model_version",
        "workflow",
        "resource_profile",
        "max_runtime_seconds",
    }
    optional_fields = {
        "max_attempts",
        "input_name",
        "output_name",
        "model_parameter_names",
        "allowed_wsi_plugin_ids",
    }
    for item in parsed:
        if (
            not isinstance(item, dict)
            or not required_fields.issubset(item)
            or not set(item).issubset(required_fields | optional_fields)
        ):
            raise ValueError(
                "each app mapping must contain: " + ", ".join(sorted(required_fields))
            )
        values = {name: item[name] for name in required_fields}
        string_fields = required_fields - {"max_runtime_seconds"}
        if any(
            not isinstance(values[name], str) or not values[name].strip()
            for name in string_fields
        ):
            raise ValueError("app mapping string values must be non-empty")
        runtime = values["max_runtime_seconds"]
        if not isinstance(runtime, int) or isinstance(runtime, bool) or runtime < 1:
            raise ValueError("max_runtime_seconds must be a positive integer")
        max_attempts = item.get("max_attempts", 3)
        if (
            not isinstance(max_attempts, int)
            or isinstance(max_attempts, bool)
            or max_attempts < 1
        ):
            raise ValueError("max_attempts must be a positive integer")
        input_name = _optional_identifier(item, "input_name", "slide")
        output_name = _optional_identifier(item, "output_name", "heatmap")
        model_parameter_names = _identifier_list(item, "model_parameter_names")
        allowed_wsi_plugin_ids = _identifier_list(item, "allowed_wsi_plugin_ids")

        mapping = AppMapping(
            app_id=values["app_id"].strip(),
            app_version=values["app_version"].strip(),
            model_id=values["model_id"].strip(),
            model_version=values["model_version"].strip(),
            workflow=values["workflow"].strip(),
            resource_profile=values["resource_profile"].strip(),
            max_runtime_seconds=runtime,
            max_attempts=max_attempts,
            input_name=input_name,
            output_name=output_name,
            model_parameter_names=model_parameter_names,
            allowed_wsi_plugin_ids=allowed_wsi_plugin_ids,
        )
        key = (mapping.app_id, mapping.app_version)
        if key in keys:
            raise ValueError(
                f"duplicate app mapping for {mapping.app_id!r} {mapping.app_version!r}"
            )
        keys.add(key)
        mappings.append(mapping)
    return tuple(mappings)


def _optional_identifier(item: dict[str, Any], name: str, default: str) -> str:
    """Read a nonempty optional mapping string or its default and strip surrounding whitespace.

    Args:
        item: Decoded mapping whose optional configuration fields are being read.
        name: Input/output name or configuration field selected by this operation.
        default: Fallback value used when the setting or optional field is absent.

    Returns:
        str: Normalized nonempty field value, or the supplied default.

    Raises:
        ValueError: A value fails the validation described above.
    """
    value = item.get(name, default)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _identifier_list(item: dict[str, Any], name: str) -> tuple[str, ...]:
    """Validate and normalize a list of unique nonempty identifier strings.

    Args:
        item: Decoded mapping whose optional configuration fields are being read.
        name: Input/output name or configuration field selected by this operation.

    Returns:
        tuple[str, ...]: Normalized unique identifiers in their configured order.

    Raises:
        ValueError: A value fails the validation described above.
    """
    value = item.get(name, [])
    if not isinstance(value, list) or any(
        not isinstance(entry, str) or not entry.strip() for entry in value
    ):
        raise ValueError(f"{name} must be a list of non-empty strings")
    normalized = tuple(entry.strip() for entry in value)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    return normalized


def _parse_storage_mappings(raw: str) -> tuple[StorageMapping, ...]:
    """Require exact storage mapping fields and construct validated mount translations from JSON.

    Args:
        raw: JSON text or environment value to parse.

    Returns:
        tuple[StorageMapping, ...]: Validated provider-prefix and worker-mount
            translations.

    Raises:
        ValueError: A value fails the validation described above.
    """
    parsed: Any = json.loads(raw)
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("EMPAIA_STORAGE_MAPPINGS_JSON must be a non-empty JSON list")
    mappings: list[StorageMapping] = []
    expected = {"storage_id", "source_prefix", "mount_root"}
    for item in parsed:
        if not isinstance(item, dict) or set(item) != expected:
            raise ValueError(
                "each storage mapping must contain storage_id, source_prefix, mount_root"
            )
        if any(not isinstance(item[name], str) for name in expected):
            raise ValueError("storage mapping values must be strings")
        mappings.append(
            StorageMapping(
                storage_id=item["storage_id"],
                source_prefix=item["source_prefix"],
                mount_root=Path(item["mount_root"]),
            )
        )
    return tuple(mappings)
