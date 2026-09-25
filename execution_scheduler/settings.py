from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class SchedulerSettings:
    database_url: str
    provider_context_key: str
    ray_serve_origin: str
    inference_serve_origin: str | None = None
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
        values = environment or os.environ
        return cls(
            database_url=_required(values, "DATABASE_URL"),
            provider_context_key=_required(values, "PROVIDER_CONTEXT_KEY"),
            ray_serve_origin=_required(values, "RAY_SERVE_ORIGIN"),
            inference_serve_origin=values.get("INFERENCE_SERVE_ORIGIN", "").strip()
            or None,
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
        if self.maximum_backoff < self.initial_backoff:
            raise ValueError("maximum backoff must not be shorter than initial backoff")
        if self.maximum_deadline < self.default_deadline:
            raise ValueError(
                "maximum deadline must not be shorter than default deadline"
            )


def _required(values: Mapping[str, str], name: str) -> str:
    value = values.get(name, "").strip()
    if not value:
        raise ValueError(f"{name} is required")
    return value


def _positive_int(values: Mapping[str, str], name: str, default: int) -> int:
    try:
        value = int(values.get(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_float(values: Mapping[str, str], name: str, default: float) -> float:
    value = _nonnegative_float(values, name, default)
    if value == 0:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_float(
    values: Mapping[str, str],
    name: str,
    default: float,
) -> float:
    try:
        value = float(values.get(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be a number") from error
    if value < 0:
        raise ValueError(f"{name} cannot be negative")
    return value
