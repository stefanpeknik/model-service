"""Read facade process settings from environment variables.

The Ray Serve origin is required; the catalog path and builder selection are
optional.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from pydantic import (
    Field,
    StringConstraints,
)
from pydantic_settings import BaseSettings, EnvSettingsSource, SettingsConfigDict


if TYPE_CHECKING:
    from collections.abc import Mapping

    from pydantic_settings import PydanticBaseSettingsSource


_RequiredString = Annotated[str, StringConstraints(min_length=1)]


class SchedulerSettings(BaseSettings):
    """Hold the catalog and Ray Serve settings for the facade process."""

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
