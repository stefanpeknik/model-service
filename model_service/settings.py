"""Read scheduler process settings from environment variables.

Required storage, encryption, and dispatch settings are separate from optional EMPAIA
and per-model origins. Parsed durations feed runtime wiring and retry policies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import timedelta
from typing import TYPE_CHECKING, Annotated

from pydantic import HttpUrl, StringConstraints, TypeAdapter


if TYPE_CHECKING:
    from collections.abc import Mapping


_ModelPrefix = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9._-]+$")]
_MODEL_ORIGINS = TypeAdapter(dict[_ModelPrefix, HttpUrl])


@dataclass(frozen=True, slots=True)
class SchedulerSettings:
    """Hold database, transport, retry, lease, capacity, and deadline settings for the public scheduler process.

    Attributes:
        database_url: PostgreSQL connection URL or libpq DSN.
        provider_context_key: URL-safe base64 AES-256 key for protected context.
        ray_serve_origin: Internal workflow dispatch and route-health HTTP origin.
        inference_serve_origin: Optional raw inference default origin; otherwise use the
            workflow origin.
        inference_model_origins: Exact model-prefix overrides for raw inference routing.
        inference_timeout: Maximum raw inference HTTP wait.
        lease_duration: Ownership lifetime extended by worker renewals.
        poll_interval: Seconds between maintenance and empty/failed claim polls.
        processing_attempts: Number of computation attempts consumed; confirmed
            preacceptance rejections restore this budget.
        provider_finish_attempts: Maximum provider publication/report attempts.
        initial_backoff: Delay after the first retryable outcome.
        maximum_backoff: Upper bound on exponential retry delay.
        default_deadline: Default end-to-end schedule budget.
        maximum_deadline: Upper bound on caller-requested schedule budget.
        per_workflow_limit: Maximum concurrent leased processing attempts per workflow.
        empaia_gateway_factory: Optional module:attribute enabling EMPAIA integration.
    """

    database_url: str
    provider_context_key: str
    ray_serve_origin: str
    inference_serve_origin: str | None = None
    inference_model_origins: Mapping[str, HttpUrl] = field(default_factory=dict)
    inference_timeout: timedelta = timedelta(minutes=15)
    lease_duration: timedelta = timedelta(seconds=60)
    poll_interval: float = 0.5
    processing_attempts: int = 3
    provider_finish_attempts: int = 3
    initial_backoff: timedelta = timedelta(seconds=5)
    maximum_backoff: timedelta = timedelta(minutes=5)
    default_deadline: timedelta = timedelta(hours=6)
    maximum_deadline: timedelta = timedelta(hours=24)
    per_workflow_limit: int = 1
    empaia_gateway_factory: str | None = None

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> SchedulerSettings:
        """Parse required and optional environment settings, validating model origins and numeric bounds.

        Args:
            environment: Optional settings mapping; environment variables are used when
                not supplied.

        Returns:
            SchedulerSettings: Parsed and validated scheduler process configuration.

        Raises:
            ValueError: A required setting is missing, a numeric setting fails its
                bounds, or maximum durations are shorter than their defaults.
            ValidationError: INFERENCE_MODEL_ORIGINS is invalid JSON or contains an
                invalid model prefix or HTTP URL.
        """
        values = environment or os.environ
        return cls(
            database_url=_required(values, "DATABASE_URL"),
            provider_context_key=_required(values, "PROVIDER_CONTEXT_KEY"),
            ray_serve_origin=_required(values, "RAY_SERVE_ORIGIN"),
            inference_serve_origin=values.get("INFERENCE_SERVE_ORIGIN", "").strip()
            or None,
            inference_model_origins=_MODEL_ORIGINS.validate_json(
                values.get("INFERENCE_MODEL_ORIGINS", "{}")
            ),
            inference_timeout=timedelta(
                seconds=_positive_float(values, "INFERENCE_TIMEOUT_SECONDS", 900)
            ),
            lease_duration=timedelta(
                seconds=_positive_float(values, "LEASE_SECONDS", 60)
            ),
            poll_interval=_positive_float(values, "WORKER_POLL_SECONDS", 0.5),
            processing_attempts=_positive_int(values, "PROCESSING_ATTEMPTS", 3),
            provider_finish_attempts=_positive_int(
                values,
                "PROVIDER_FINISH_ATTEMPTS",
                3,
            ),
            initial_backoff=timedelta(
                seconds=_nonnegative_float(values, "INITIAL_BACKOFF_SECONDS", 5)
            ),
            maximum_backoff=timedelta(
                seconds=_nonnegative_float(values, "MAXIMUM_BACKOFF_SECONDS", 300)
            ),
            default_deadline=timedelta(
                seconds=_positive_float(values, "DEFAULT_DEADLINE_SECONDS", 21_600)
            ),
            maximum_deadline=timedelta(
                seconds=_positive_float(values, "MAXIMUM_DEADLINE_SECONDS", 86_400)
            ),
            per_workflow_limit=_positive_int(values, "PER_WORKFLOW_LIMIT", 1),
            empaia_gateway_factory=values.get("EMPAIA_GATEWAY_FACTORY") or None,
        )

    def __post_init__(self) -> None:
        """Reject maximum backoff or deadline values shorter than their corresponding initial/default values.

        Raises:
            ValueError: maximum_backoff is shorter than initial_backoff, or
                maximum_deadline is shorter than default_deadline.
        """
        if self.maximum_backoff < self.initial_backoff:
            raise ValueError("maximum backoff must not be shorter than initial backoff")
        if self.maximum_deadline < self.default_deadline:
            raise ValueError(
                "maximum deadline must not be shorter than default deadline"
            )


def _required(values: Mapping[str, str], name: str) -> str:
    """Return a nonempty stripped environment value or raise ValueError naming the missing setting.

    Args:
        values: Environment variable names mapped to string settings.
        name: Required environment setting whose value must be nonempty after stripping.

    Returns:
        str: Nonempty stripped environment setting.

    Raises:
        ValueError: The named setting is missing or empty after stripping whitespace.
    """
    value = values.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _positive_int(values: Mapping[str, str], name: str, default: int) -> int:
    """Parse a positive integer setting or its default, raising a setting-specific validation error.

    Args:
        values: Environment variable names mapped to string settings.
        name: Environment variable name to parse.
        default: Fallback value used when the setting or optional field is absent.

    Returns:
        int: Parsed setting, or default when absent, validated as greater than zero.

    Raises:
        ValueError: The named setting cannot be parsed as an integer or is nonpositive.
    """
    try:
        value = int(values.get(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_float(values: Mapping[str, str], name: str, default: float) -> float:
    """Parse a numeric setting greater than zero using the shared nonnegative parser.

    Args:
        values: Environment variable names mapped to string settings.
        name: Environment variable name to parse.
        default: Fallback value used when the setting or optional field is absent.

    Returns:
        float: Parsed setting, or default when absent, validated as greater than zero.

    Raises:
        ValueError: The named setting cannot be parsed as a number or is nonpositive.
    """
    value = _nonnegative_float(values, name, default)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_float(
    values: Mapping[str, str],
    name: str,
    default: float,
) -> float:
    """Parse a numeric setting or default and reject negative values.

    Args:
        values: Environment variable names mapped to string settings.
        name: Environment variable name to parse.
        default: Fallback value used when the setting or optional field is absent.

    Returns:
        float: Parsed setting, or default when absent, validated as nonnegative.

    Raises:
        ValueError: The named setting cannot be parsed as a number or is negative.
    """
    try:
        value = float(values.get(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
    return value
