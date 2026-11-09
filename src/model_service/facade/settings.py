"""Read scheduler process settings from environment variables.

Required storage, encryption, and dispatch settings are separate from optional
per-model origins; each provider reads its own settings. Parsed durations feed runtime wiring and retry policies.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
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
    """Accept typed durations or finite numeric seconds for duration validation.

    Args:
        value: A `timedelta` or a number of seconds.

    Returns:
        The `timedelta` unchanged, or the seconds as a float.

    Raises:
        ValidationError: If the value is not a finite number.
    """
    return value if isinstance(value, timedelta) else _SECONDS.validate_python(value)


_SecondsDuration = Annotated[timedelta, BeforeValidator(_duration_from_seconds)]
_RequiredString = Annotated[str, StringConstraints(min_length=1)]


class SchedulerSettings(BaseSettings):
    """Hold database, transport, retry, lease, capacity, and deadline settings for the public scheduler process."""

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
    ray_dashboard_origin: str | None = Field(
        default=None, validation_alias="RAY_DASHBOARD_ORIGIN"
    )
    capacity_grace: _SecondsDuration = Field(
        default=timedelta(seconds=60),
        ge=timedelta(0),
        validation_alias="CAPACITY_GRACE_SECONDS",
    )
    model_catalog_path: Path = Field(
        default=Path("/etc/model-service/model-catalog.json"),
        validation_alias="MODEL_CATALOG_PATH",
    )
    builder_ids: frozenset[_RequiredString] | None = Field(
        default=None, min_length=1, validation_alias="BUILDER_IDS"
    )
    shared_storage_roots: tuple[Path, ...] = Field(
        default=(), validation_alias="SHARED_STORAGE_ROOTS"
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
    per_model_limit: int = Field(default=1, gt=0, validation_alias="PER_MODEL_LIMIT")
    max_concurrent_executions: int = Field(
        default=4, gt=0, validation_alias="MAX_CONCURRENT_EXECUTIONS"
    )
    per_model_inference_limit: int = Field(
        default=32, gt=0, validation_alias="PER_MODEL_INFERENCE_LIMIT"
    )
    max_concurrent_inference: int = Field(
        default=32, gt=0, validation_alias="MAX_CONCURRENT_INFERENCE"
    )
    inference_max_request_bytes: int = Field(
        default=2 * 1024 * 1024, gt=0, validation_alias="INFERENCE_MAX_REQUEST_BYTES"
    )
    inference_max_result_bytes: int = Field(
        default=4 * 1024 * 1024, gt=0, validation_alias="INFERENCE_MAX_RESULT_BYTES"
    )
    inference_retention: _SecondsDuration = Field(
        default=timedelta(hours=24),
        gt=timedelta(0),
        validation_alias="INFERENCE_RETENTION_SECONDS",
    )
    # Request and response files of scheduled raw inference; every facade
    # replica must see the same directory.
    inference_storage_path: Path = Field(
        default=Path("/mnt/model-service-inference"),
        validation_alias="INFERENCE_STORAGE_PATH",
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
        """Read settings only from constructor arguments.

        Direct construction stays isolated from the environment; `from_environment`
        selects the environment source.

        Args:
            settings_cls: The settings class.
            init_settings: Constructor arguments.
            env_settings: Environment variables; unused.
            dotenv_settings: The dotenv file; unused.
            file_secret_settings: Secret files; unused.

        Returns:
            Only the constructor source.
        """
        return (init_settings,)

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> SchedulerSettings:
        """Parse settings from environment variables.

        Model origins and numeric bounds are validated.

        Args:
            environment: Variables to read instead of the process environment.

        Returns:
            The settings.

        Raises:
            ValidationError: If a required setting is missing or a value is invalid.
        """
        source = EnvSettingsSource(cls)
        if environment is not None:
            source.env_vars = dict(environment)
        return cls(**source())

    @field_validator(
        "inference_serve_origin",
        "ray_dashboard_origin",
        mode="before",
    )
    @classmethod
    def _empty_optional_setting(cls, value: object) -> object:
        """Treat blank optional origins as absent.

        Args:
            value: The raw setting.

        Returns:
            None for a blank string, the stripped string, or the value unchanged.
        """
        if isinstance(value, str):
            return value.strip() or None
        return value

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
