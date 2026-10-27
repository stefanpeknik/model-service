"""EMPAIA wiring for the processes that talk to App Service, from `EMPAIA_*` variables.

The facade and the builder replicas read the same settings: the facade admits
jobs and publishes results, builders read job inputs.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import Field, SecretStr, model_validator
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

    # Marketplace that the facade registers the deployed models' apps in at
    # startup; none, no registration. `jes_target` is model-service's name in
    # JES_HTTP_TARGETS, part of each app's image.
    marketplace_url: str | None = None
    organization_id: str | None = None
    jes_target: str = "model-service"
    # Client credentials for the marketplace; without them, no Authorization.
    idp_url: str | None = None
    client_id: str | None = None
    client_secret: SecretStr | None = None

    @model_validator(mode="after")
    def _complete(self) -> Self:
        if self.marketplace_url is not None and not self.organization_id:
            raise ValueError("EMPAIA_ORGANIZATION_ID is required with a marketplace")
        if self.idp_url is not None and (
            not self.client_id or self.client_secret is None
        ):
            raise ValueError(
                "EMPAIA_CLIENT_ID and EMPAIA_CLIENT_SECRET go with EMPAIA_IDP_URL"
            )
        return self
