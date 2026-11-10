"""EMPAIA wiring for the processes that talk to App Service, from `EMPAIA_*` variables.

The API and the builder replicas read the same settings: the API admits
jobs and publishes results, builders read job inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Self

from pydantic import Field, PrivateAttr, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


@dataclass(frozen=True, slots=True)
class MarketplaceClient:
    """Client credentials for the marketplace's identity provider."""

    idp_url: str
    client_id: str
    client_secret: SecretStr


@dataclass(frozen=True, slots=True)
class MarketplaceTarget:
    """Where and as whom the API registers apps."""

    url: str
    organization_id: str
    jes_target: str
    # None: requests carry no Authorization.
    client: MarketplaceClient | None


class EmpaiaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EMPAIA_", frozen=True, extra="ignore")

    # AppService URLs that may receive EMPAIA job credentials; empty refuses all.
    api_urls: tuple[str, ...] = ()
    workspace_root: Path = Path("/mnt/model-service-executions")
    # Creating a pixelmap took 10 s on the EMPAIA test instance; the SDK's 5 s
    # default turned that into a failed job with an orphaned pixelmap.
    app_service_timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)

    # Marketplace that the API registers the deployed models' apps in at
    # startup; none, no registration. `jes_target` is model-service's name in
    # JES_HTTP_TARGETS, part of each app's image.
    marketplace_url: str | None = None
    organization_id: str | None = None
    jes_target: str = "model-service"
    # Client credentials for the marketplace; without them, no Authorization.
    idp_url: str | None = None
    client_id: str | None = None
    client_secret: SecretStr | None = None
    _marketplace: MarketplaceTarget | None = PrivateAttr(default=None)

    @property
    def marketplace(self) -> MarketplaceTarget | None:
        """The marketplace to register apps in, or None for no registration."""
        return self._marketplace

    @model_validator(mode="after")
    def _complete(self) -> Self:
        """Check that settings that go together are set together, and combine them.

        Returns:
            The settings, with `marketplace` set when a marketplace URL is.

        Raises:
            ValueError: If a marketplace URL has no organization ID, or an identity
                provider URL lacks the client ID or secret.
        """
        client = None
        if self.idp_url is not None:
            if not self.client_id or self.client_secret is None:
                raise ValueError(
                    "EMPAIA_CLIENT_ID and EMPAIA_CLIENT_SECRET go with EMPAIA_IDP_URL"
                )
            client = MarketplaceClient(self.idp_url, self.client_id, self.client_secret)
        if self.marketplace_url is not None:
            if not self.organization_id:
                raise ValueError(
                    "EMPAIA_ORGANIZATION_ID is required with a marketplace"
                )
            self._marketplace = MarketplaceTarget(
                self.marketplace_url, self.organization_id, self.jes_target, client
            )
        return self
