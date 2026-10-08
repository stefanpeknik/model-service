"""Read scheduler process settings from environment variables.

Required storage, encryption, and dispatch settings are separate from optional EMPAIA
and per-model origins. Parsed durations feed runtime wiring and retry policies.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING, Annotated, Self

from pydantic import (
    BeforeValidator,
    Field,
    FiniteFloat,
    HttpUrl,
    StringConstraints,
    TypeAdapter,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, EnvSettingsSource, SettingsConfigDict


if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic_settings import PydanticBaseSettingsSource


_ModelPrefix = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9._-]+$")]
_SECONDS = TypeAdapter(FiniteFloat)


def _duration_from_seconds(value: object) -> timedelta | float:
    """Accept typed durations or finite numeric seconds for duration validation."""
    return value if isinstance(value, timedelta) else _SECONDS.validate_python(value)


_SecondsDuration = Annotated[timedelta, BeforeValidator(_duration_from_seconds)]
_RequiredString = Annotated[str, StringConstraints(min_length=1)]


class SchedulerSettings(BaseSettings):
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

    model_config = SettingsConfigDict(
        frozen=True,
        case_sensitive=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        hide_input_in_errors=True,
    )

    database_url: _RequiredString = Field(validation_alias="DATABASE_URL", repr=False)
    provider_context_key: _RequiredString = Field(
        validation_alias="PROVIDER_CONTEXT_KEY", repr=False
    )
    ray_serve_origin: _RequiredString = Field(validation_alias="RAY_SERVE_ORIGIN")
    workflow_ids: frozenset[_RequiredString] | None = Field(
        default=None, min_length=1, validation_alias="WORKFLOW_IDS"
    )
    inference_serve_origin: str | None = Field(
        default=None, validation_alias="INFERENCE_SERVE_ORIGIN"
    )
    inference_model_origins: dict[_ModelPrefix, HttpUrl] = Field(
        default_factory=dict, validation_alias="INFERENCE_MODEL_ORIGINS"
    )
    inference_timeout: _SecondsDuration = Field(
        default=timedelta(minutes=15),
        gt=timedelta(0),
        validation_alias="INFERENCE_TIMEOUT_SECONDS",
    )
    lease_duration: _SecondsDuration = Field(
        default=timedelta(seconds=60), gt=timedelta(0), validation_alias="LEASE_SECONDS"
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
    per_workflow_limit: int = Field(
        default=1, gt=0, validation_alias="PER_WORKFLOW_LIMIT"
    )
    empaia_gateway_factory: str | None = Field(
        default=None, validation_alias="EMPAIA_GATEWAY_FACTORY"
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Keep direct construction isolated; from_environment selects the environment source."""
        return (init_settings,)

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
            SettingsError: INFERENCE_MODEL_ORIGINS cannot be decoded as JSON.
            ValidationError: Required values, numeric bounds, model origins, or
                cross-field policy constraints are invalid.
        """
        source = EnvSettingsSource(cls)
        if environment is not None:
            source.env_vars = dict(environment)
        return cls(**source())

    @field_validator("inference_serve_origin", "empaia_gateway_factory", mode="before")
    @classmethod
    def _empty_optional_setting(cls, value: object) -> object:
        """Treat blank optional origins and factory settings as absent."""
        if isinstance(value, str):
            return value.strip() or None
        return value

    @model_validator(mode="after")
    def _validate_policy_bounds(self) -> Self:
        """Reject maximum backoff or deadline values shorter than their corresponding initial/default values.

        Returns:
            Self: Validated settings with consistent backoff and deadline limits.

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
        return self
