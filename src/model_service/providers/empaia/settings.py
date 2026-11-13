"""EMPAIA wiring for the processes that talk to App Service, from `EMPAIA_*` variables.

The API and the builder replicas read the same settings: the API admits
jobs and publishes results, builders read job inputs.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmpaiaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EMPAIA_", frozen=True, extra="ignore")

    # AppService URLs that may receive EMPAIA job credentials; empty refuses all.
    api_urls: tuple[str, ...] = ()
    workspace_root: Path = Path("/mnt/model-service-executions")
    # App Service requests; the SDK's 5 s default is too short for some writes.
    app_service_timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)
