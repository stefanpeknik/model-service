"""EMPAIA wiring for the processes that talk to App Service, from `EMPAIA_*` variables.

The facade and the builder replicas read the same settings: the facade admits
jobs and publishes results, builders read job inputs.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SlideAccess(StrEnum):
    """How slide inputs reach the builder."""

    DIRECT = "direct"  # link the file named by the RationAI direct-access route
    HTTP = "http"  # download the needed level through the App API's region route


class EmpaiaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EMPAIA_", frozen=True, extra="ignore")

    # AppService URLs that may receive EMPAIA job credentials; empty refuses all.
    api_urls: tuple[str, ...] = ()
    workspace_root: Path = Path("/mnt/model-service-executions")
    slide_access: SlideAccess = SlideAccess.DIRECT
    # Creating a pixelmap took 10 s on the EMPAIA test instance; the SDK's 5 s
    # default turned that into a failed job with an orphaned pixelmap.
    app_service_timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)
