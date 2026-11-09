"""Read facade process settings from environment variables.

The Ray Serve origin is required; the catalog path, builder selection and raw
inference origins, timeout and size limits are optional.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from pydantic import (
    BeforeValidator,
    Field,
    FiniteFloat,
    HttpUrl,
    StringConstraints,
    TypeAdapter,
    field_validator,
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
    """Hold catalog, dispatch and raw inference settings for the facade process."""

    model_config = SettingsConfigDict(
        frozen=True,
        case_sensitive=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        hide_input_in_errors=True,
    )

    ray_serve_origin: _RequiredString = Field(validation_alias="RAY_SERVE_ORIGIN")
    model_catalog_path: Path = Field(
        default=Path("/etc/model-service/model-catalog.json"),
        validation_alias="MODEL_CATALOG_PATH",
    )
    builder_ids: frozenset[_RequiredString] | None = Field(
        default=None, min_length=1, validation_alias="BUILDER_IDS"
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
    inference_max_request_bytes: int = Field(
        default=2 * 1024 * 1024, gt=0, validation_alias="INFERENCE_MAX_REQUEST_BYTES"
    )
    inference_max_result_bytes: int = Field(
        default=4 * 1024 * 1024, gt=0, validation_alias="INFERENCE_MAX_RESULT_BYTES"
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
        mode="before",
    )
    @classmethod
    def _empty_optional_setting(cls, value: object) -> object:
        """Treat a blank optional origin as absent.

        Args:
            value: The raw setting.

        Returns:
            None for a blank string, the stripped string, or the value unchanged.
        """
        if isinstance(value, str):
            return value.strip() or None
        return value
