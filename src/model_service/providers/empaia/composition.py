"""The EMPAIA provider, composed for the facade and for builder replicas.

It serves JES's remote execution routes, runs App Service jobs, and registers
apps for the models whose files have an `empaia` block. Its settings come from
`EMPAIA_*` (`EmpaiaSettings`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_service.providers.composition import FacadeContribution
from model_service.providers.empaia.gateway import INPUT_ACCESS_KIND
from model_service.providers.empaia.http_gateway import http_gateway_factory
from model_service.providers.empaia.settings import EmpaiaSettings


if TYPE_CHECKING:
    from collections.abc import Mapping

    from model_service.builders.application import AccessAdapter
    from model_service.builders.settings import WorkerSettings
    from model_service.facade.catalog import ModelCatalog
    from model_service.facade.settings import SchedulerSettings
    from model_service.providers.empaia.registration import Marketplace


# The model files' block it reads: how a model is offered in EMPAIA.
CATALOG_BLOCK = "empaia"


class EmpaiaProvider:
    catalog_block = CATALOG_BLOCK

    def facade(
        self, catalog: ModelCatalog, settings: SchedulerSettings
    ) -> FacadeContribution:
        from model_service.providers.empaia.apps import app_definitions, parse_listings
        from model_service.providers.empaia.provider import EmpaiaProviderAdapter
        from model_service.providers.empaia.publishers import default_publishers
        from model_service.providers.empaia.registration import register_until_done
        from model_service.providers.empaia.routes import create_router

        del settings
        empaia = EmpaiaSettings()
        publishers = default_publishers()
        apps = app_definitions(
            catalog, parse_listings(catalog.provider_block(CATALOG_BLOCK)), publishers
        )
        marketplace = _marketplace(empaia)

        async def register() -> None:
            if marketplace is not None and apps:
                await register_until_done(marketplace, apps)

        return FacadeContribution(
            adapter=EmpaiaProviderAdapter(
                http_gateway_factory(empaia),
                allowed_api_urls=empaia.api_urls,
                workspace_root=empaia.workspace_root,
                publishers=publishers,
            ),
            routes=create_router,
            # In the background: the API serves while the marketplace is down.
            background=register,
        )

    def input_access(self, settings: WorkerSettings) -> Mapping[str, AccessAdapter]:
        from model_service.providers.empaia.input_access import (
            EmpaiaInputAccessAdapter,
        )

        del settings
        empaia = EmpaiaSettings()
        return {
            INPUT_ACCESS_KIND: EmpaiaInputAccessAdapter(
                http_gateway_factory(empaia), empaia.workspace_root
            )
        }


def _marketplace(settings: EmpaiaSettings) -> Marketplace | None:
    from model_service.providers.empaia.credentials import (
        ClientCredentials,
        NoCredentials,
        OutgoingCredentials,
    )
    from model_service.providers.empaia.registration import Marketplace

    if settings.marketplace_url is None or settings.organization_id is None:
        return None
    credentials: OutgoingCredentials = NoCredentials()
    if settings.idp_url is not None:
        assert settings.client_id is not None and settings.client_secret is not None
        credentials = ClientCredentials(
            settings.idp_url,
            settings.client_id,
            settings.client_secret.get_secret_value(),
        )
    return Marketplace(
        settings.marketplace_url,
        settings.organization_id,
        settings.jes_target,
        credentials,
        timeout=settings.app_service_timeout_seconds,
    )


PROVIDER = EmpaiaProvider()
