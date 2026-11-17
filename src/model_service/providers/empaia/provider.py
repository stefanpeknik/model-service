"""The EMPAIA provider, composed for the API and for builder replicas.

It serves JES's remote execution routes, runs App Service jobs, and registers
apps, with their preprocessing triggers, for the models whose files have an
`empaia` block. Its settings come from `EMPAIA_*` (`EmpaiaSettings`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_service.providers.empaia.app_service.gateway import INPUT_ACCESS_KIND
from model_service.providers.empaia.app_service.http import http_gateway_factory
from model_service.providers.empaia.settings import EmpaiaSettings
from model_service.providers.provider import ApiContribution, Provider


if TYPE_CHECKING:
    from collections.abc import Mapping

    from model_service.api.model_catalog import ModelCatalog
    from model_service.api.scheduler.settings import SchedulerSettings
    from model_service.builders.base import ReplicaSettings
    from model_service.builders.input_access import InputAccessFactory
    from model_service.providers.empaia.credentials import OutgoingCredentials
    from model_service.providers.empaia.registration import Marketplace
    from model_service.providers.empaia.triggers import PreprocessingTriggers


# The model files' block it reads: how a model is offered in EMPAIA.
CATALOG_BLOCK = "empaia"


class EmpaiaProvider(Provider):
    """Runs EMPAIA jobs and registers the deployed models' EMPAIA apps."""

    catalog_block = CATALOG_BLOCK

    def api(
        self, catalog: ModelCatalog, settings: SchedulerSettings
    ) -> ApiContribution:
        """Contribute the EMPAIA scheduler adapter, JES routes and app registration.

        Apps come from the models whose file has an `empaia` block. Registration runs
        in the background, and only when a marketplace is configured, so the API
        serves while the marketplace is down; it aligns the preprocessing
        triggers too when a Medical Data Service is configured.

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
        credentials = _credentials(empaia)
        marketplace = _marketplace(empaia, credentials)
        triggers = _triggers(empaia, credentials)

        async def register() -> None:
            """Register the apps until done, when a marketplace is configured.

            It runs with no apps too, so triggers of removed models go.
            """
            if marketplace is not None:
                await register_until_done(marketplace, apps, triggers)

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


def _credentials(settings: EmpaiaSettings) -> OutgoingCredentials:
    """Build the credentials model-service registers apps and triggers with.

    Args:
        settings: The EMPAIA settings.

    Returns:
        Client credentials when an identity provider is set, otherwise none.
    """
    from model_service.providers.empaia.credentials import (
        ClientCredentials,
        NoCredentials,
    )

    target = settings.marketplace
    if target is None or target.client is None:
        return NoCredentials()
    return ClientCredentials(
        target.client.idp_url,
        target.client.client_id,
        target.client.client_secret.get_secret_value(),
    )


def _marketplace(
    settings: EmpaiaSettings, credentials: OutgoingCredentials
) -> Marketplace | None:
    """Build the marketplace client the settings describe.

    Args:
        settings: The EMPAIA settings.
        credentials: Supplies the `Authorization` header.

    Returns:
        The marketplace, or None without a marketplace URL and organization ID.
    """
    from model_service.providers.empaia.registration import Marketplace

    target = settings.marketplace
    if target is None:
        return None
    return Marketplace(
        target.url,
        target.organization_id,
        target.jes_target,
        credentials,
        timeout=settings.app_service_timeout_seconds,
    )


def _triggers(
    settings: EmpaiaSettings, credentials: OutgoingCredentials
) -> PreprocessingTriggers | None:
    """Build the preprocessing trigger client the settings describe.

    Triggers it creates are owned by `model-service:<jes_target>`.

    Args:
        settings: The EMPAIA settings.
        credentials: Supplies the `Authorization` header.

    Returns:
        The triggers, or None without a Medical Data Service URL.
    """
    from model_service.providers.empaia.triggers import PreprocessingTriggers

    target = settings.marketplace
    if target is None or target.mds_url is None:
        return None
    return PreprocessingTriggers(
        target.mds_url,
        f"model-service:{target.jes_target}",
        credentials,
        timeout=settings.app_service_timeout_seconds,
    )


PROVIDER = EmpaiaProvider()
