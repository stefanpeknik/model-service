"""The EMPAIA provider, composed for the API and for builder replicas.

It serves JES's remote execution routes, runs App Service jobs, and registers
apps for the models whose files have an `empaia` block. Its settings come from
`EMPAIA_*` (`EmpaiaSettings`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_service.providers.empaia.app_service.gateway import INPUT_ACCESS_KIND
from model_service.providers.empaia.app_service.http import http_gateway_factory
from model_service.providers.empaia.settings import EmpaiaSettings
from model_service.providers.provider import ApiContribution, Provider


if TYPE_CHECKING:
    from collections.abc import Mapping

    from model_service.builders.base import ReplicaSettings
    from model_service.builders.input_access import InputAccessFactory
    from model_service.deployed_models.catalog import ModelCatalog
    from model_service.providers.empaia.registration import Marketplace
    from model_service.scheduler.settings import SchedulerSettings


# The model files' block it reads: how a model is offered in EMPAIA.
CATALOG_BLOCK = "empaia"


class EmpaiaProvider(Provider):
    catalog_block = CATALOG_BLOCK

    def api(
        self, catalog: ModelCatalog, settings: SchedulerSettings
    ) -> ApiContribution:
        """Contribute the EMPAIA scheduler adapter, JES routes and app registration.

        Apps come from the models whose file has an `empaia` block. Registration runs
        in the background, and only when a marketplace is configured, so the API
        serves while the marketplace is down.

        Args:
            catalog: The deployed models and their `empaia` blocks.
            settings: Unused; EMPAIA reads its own `EMPAIA_*` settings.

        Returns:
            the API contribution.

        Raises:
            pydantic.ValidationError: If the `EMPAIA_*` settings are invalid.
            ValueError: If a model's listing is invalid or an app's EAD references an
                unknown input.
            jsonschema.ValidationError: If a generated EAD fails the EAD v3 schema.
        """
        from model_service.providers.empaia.ead.apps import (
            app_definitions,
            parse_listings,
        )
        from model_service.providers.empaia.jobs.adapter import EmpaiaProviderAdapter
        from model_service.providers.empaia.jobs.routes import create_router
        from model_service.providers.empaia.publishing.publishers import (
            default_publishers,
        )
        from model_service.providers.empaia.registration import register_until_done

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

        return ApiContribution(
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

    def input_access(
        self, settings: ReplicaSettings
    ) -> Mapping[str, InputAccessFactory]:
        """Give builder replicas access to EMPAIA job inputs through the App Service.

        Args:
            settings: Unused; EMPAIA reads its own `EMPAIA_*` settings.

        Returns:
            The EMPAIA input access adapter, keyed by `INPUT_ACCESS_KIND`.

        Raises:
            pydantic.ValidationError: If the `EMPAIA_*` settings are invalid.
        """
        from model_service.providers.empaia.jobs.inputs import EmpaiaInputAccessFactory

        del settings
        empaia = EmpaiaSettings()
        return {
            INPUT_ACCESS_KIND: EmpaiaInputAccessFactory(
                http_gateway_factory(empaia), empaia.workspace_root
            )
        }


def _marketplace(settings: EmpaiaSettings) -> Marketplace | None:
    """Build the marketplace client the settings describe.

    It uses client credentials when an identity provider is set, otherwise none.

    Args:
        settings: The EMPAIA settings.

    Returns:
        The marketplace, or None without a marketplace URL and organization ID.
    """
    from model_service.providers.empaia.credentials import (
        ClientCredentials,
        NoCredentials,
        OutgoingCredentials,
    )
    from model_service.providers.empaia.registration import Marketplace

    target = settings.marketplace
    if target is None:
        return None
    credentials: OutgoingCredentials = NoCredentials()
    if target.client is not None:
        credentials = ClientCredentials(
            target.client.idp_url,
            target.client.client_id,
            target.client.client_secret.get_secret_value(),
        )
    return Marketplace(
        target.url,
        target.organization_id,
        target.jes_target,
        credentials,
        timeout=settings.app_service_timeout_seconds,
    )


PROVIDER = EmpaiaProvider()
