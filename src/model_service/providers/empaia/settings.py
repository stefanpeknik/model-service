"""EMPAIA wiring for the processes that talk to App Service, from `EMPAIA_*` variables."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmpaiaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EMPAIA_", frozen=True, extra="ignore")

    # App Service requests; the SDK's 5 s default is too short for some writes.
    app_service_timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)
