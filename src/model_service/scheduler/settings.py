"""Settings of the API process, which also runs the scheduler, from the environment.

Storage, encryption, and dispatch settings; each provider reads its own settings. Parsed durations feed runtime wiring and retry policies.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Annotated, Self

from pydantic import (
    BeforeValidator,
    Field,
    FiniteFloat,
    StringConstraints,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


# Environment values are strings; pydantic parses only numbers as seconds.
_SecondsDuration = Annotated[
    timedelta, BeforeValidator(lambda v: float(v) if isinstance(v, str) else v)
]
_RequiredString = Annotated[str, StringConstraints(min_length=1)]


class SchedulerSettings(BaseSettings):
    """Hold database, transport, retry, lease, and deadline settings for the public scheduler process."""

    model_config = SettingsConfigDict(
        frozen=True,
        case_sensitive=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        hide_input_in_errors=True,
    )

    database_url: _RequiredString = Field(validation_alias="DATABASE_URL", repr=False)
    database_pool_size: int = Field(
        default=20, gt=0, validation_alias="DATABASE_POOL_SIZE"
    )
    provider_context_key: _RequiredString = Field(
        validation_alias="PROVIDER_CONTEXT_KEY", repr=False
    )
    ray_serve_origin: _RequiredString = Field(validation_alias="RAY_SERVE_ORIGIN")
    model_catalog_path: Path = Field(
        default=Path("/etc/model-service/model-catalog.json"),
        validation_alias="MODEL_CATALOG_PATH",
    )
    # Deployed builders; unset means all registered, empty means none.
    builder_ids: frozenset[_RequiredString] | None = Field(
        default=None, validation_alias="BUILDER_IDS"
    )
    shared_storage_roots: tuple[Path, ...] = Field(
        default=(), validation_alias="SHARED_STORAGE_ROOTS"
    )
    lease_duration: _SecondsDuration = Field(
        default=timedelta(seconds=300),
        gt=timedelta(0),
        validation_alias="LEASE_SECONDS",
    )
    poll_interval: FiniteFloat = Field(
        default=0.5, gt=0, validation_alias="WORKER_POLL_SECONDS"
    )
    processing_attempts: int = Field(
        default=3, gt=0, validation_alias="PROCESSING_ATTEMPTS"
    )
    provider_finish_attempts: int = Field(
        default=3, gt=0, validation_alias="PROVIDER_FINISH_ATTEMPTS"
    )
    initial_backoff: _SecondsDuration = Field(
        default=timedelta(seconds=5),
        ge=timedelta(0),
        validation_alias="INITIAL_BACKOFF_SECONDS",
    )
    maximum_backoff: _SecondsDuration = Field(
        default=timedelta(minutes=5),
        ge=timedelta(0),
        validation_alias="MAXIMUM_BACKOFF_SECONDS",
    )
    default_deadline: _SecondsDuration = Field(
        default=timedelta(hours=6),
        gt=timedelta(0),
        validation_alias="DEFAULT_DEADLINE_SECONDS",
    )
    maximum_deadline: _SecondsDuration = Field(
        default=timedelta(hours=24),
        gt=timedelta(0),
        validation_alias="MAXIMUM_DEADLINE_SECONDS",
    )
    per_model_limit: int = Field(default=1, gt=0, validation_alias="PER_MODEL_LIMIT")
    max_concurrent_executions: int = Field(
        default=4, gt=0, validation_alias="MAX_CONCURRENT_EXECUTIONS"
    )

    @model_validator(mode="after")
    def _validate_policy_bounds(self) -> Self:
        """Reject a maximum backoff or deadline shorter than its initial or default value.

        Returns:
            The settings unchanged.

        Raises:
            ValueError: If maximum backoff is shorter than initial backoff, or
                maximum deadline is shorter than default deadline.
        """
        if self.maximum_backoff < self.initial_backoff:
            raise ValueError("maximum backoff must not be shorter than initial backoff")
        if self.maximum_deadline < self.default_deadline:
            raise ValueError(
                "maximum deadline must not be shorter than default deadline"
            )
        return self
