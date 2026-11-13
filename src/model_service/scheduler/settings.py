"""Settings of the API process, from the environment."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field, StringConstraints
from pydantic_settings import BaseSettings, SettingsConfigDict


_RequiredString = Annotated[str, StringConstraints(min_length=1)]


class SchedulerSettings(BaseSettings):
    """Hold where the API finds Ray Serve, the model catalog and the deployed builders."""

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
    # Deployed builders; unset means all registered, empty means none.
    builder_ids: frozenset[_RequiredString] | None = Field(
        default=None, validation_alias="BUILDER_IDS"
    )
