"""EMPAIA wiring for the processes that talk to App Service, from `EMPAIA_*` variables.

Builder replicas read the gateway factory and workspace root; every process that
builds a gateway reads slide access and the App Service timeout.
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

    gateway_factory: str | None = None
    workspace_root: Path = Path("/mnt/model-service-executions")
    slide_access: SlideAccess = SlideAccess.DIRECT
    # Creating a pixelmap took 10 s on the EMPAIA test instance; the SDK's 5 s
    # default turned that into a failed job with an orphaned pixelmap.
    app_service_timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)
