"""EMPAIA wiring for the processes that talk to App Service, from `EMPAIA_*` variables.

The control plane and the analysis runner read the same settings: the control plane admits
jobs and publishes results, the runner reads job inputs.
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
    """Where and as whom the control plane registers apps."""

    url: str
    organization_id: str
    jes_target: str
    # None: requests carry no Authorization.
    client: MarketplaceClient | None
    # None: preprocessing triggers are left alone.
    mds_url: str | None = None


class EmpaiaSettings(BaseSettings):
    """EMPAIA settings, read from `EMPAIA_*` environment variables.

    Attributes:
        api_urls: App Service URLs that may receive job credentials; empty
            refuses all.
        workspace_root: Where EMPAIA jobs' inputs and outputs are kept.
        app_service_timeout_seconds: Timeout of App Service calls.
        marketplace_url: Marketplace to register apps in at startup; None
            for no registration.
        organization_id: The organization that owns the apps.
        jes_target: model-service's name in JES's HTTP targets.
        mds_url: Medical Data Service whose preprocessing triggers follow the
            registered apps; None leaves triggers alone.
        idp_url: Identity provider for marketplace client credentials.
        client_id: The OAuth client id.
        client_secret: The OAuth client secret.
    """

    model_config = SettingsConfigDict(env_prefix="EMPAIA_", frozen=True, extra="ignore")

    # AppService URLs that may receive EMPAIA job credentials; empty refuses all.
    api_urls: tuple[str, ...] = ()
    workspace_root: Path = Path("/mnt/model-service-executions")
    # Creating a pixelmap took 10 s on the EMPAIA test instance; the SDK's 5 s
    # default turned that into a failed job with an orphaned pixelmap.
    app_service_timeout_seconds: float = Field(default=60, gt=0, allow_inf_nan=False)

    # Marketplace that the control plane registers the deployed models' apps in at
    # startup; none, no registration. `jes_target` is model-service's name in
    # JES_HTTP_TARGETS, part of each app's image.
    marketplace_url: str | None = None
    organization_id: str | None = None
    jes_target: str = "model-service"
    # Medical Data Service whose preprocessing triggers follow the registered
    # apps; it takes the marketplace's client credentials.
    mds_url: str | None = None
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
            ValueError: If a marketplace URL has no organization ID, a Medical
                Data Service URL has no marketplace, or an identity provider URL
                lacks the client ID or secret.
        """
        client = None
        if self.idp_url is not None:
            if not self.client_id or self.client_secret is None:
                raise ValueError(
                    "EMPAIA_CLIENT_ID and EMPAIA_CLIENT_SECRET go with EMPAIA_IDP_URL"
                )
            client = MarketplaceClient(self.idp_url, self.client_id, self.client_secret)
        if self.mds_url is not None and self.marketplace_url is None:
            raise ValueError("EMPAIA_MDS_URL goes with EMPAIA_MARKETPLACE_URL")
        if self.marketplace_url is not None:
            if not self.organization_id:
                raise ValueError(
                    "EMPAIA_ORGANIZATION_ID is required with a marketplace"
                )
            self._marketplace = MarketplaceTarget(
                self.marketplace_url,
                self.organization_id,
                self.jes_target,
                client,
                self.mds_url,
            )
        return self
