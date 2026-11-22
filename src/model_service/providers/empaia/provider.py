"""The EMPAIA provider, composed for the API and for builder replicas.

It serves JES's remote execution routes, runs App Service jobs, and registers
apps, with their preprocessing triggers, for the analyses with an
`expose.empaia` section. Its settings come from `EMPAIA_*` (`EmpaiaSettings`).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from model_service.providers.empaia.app_service.gateway import INPUT_ACCESS_KIND
from model_service.providers.empaia.app_service.http import http_gateway_factory
from model_service.providers.empaia.settings import EmpaiaSettings
from model_service.providers.provider import ApiContribution, Provider


if TYPE_CHECKING:
    from collections.abc import Mapping

    from model_service.api.model_catalog import ModelCatalogView
    from model_service.api.scheduler.settings import SchedulerSettings
    from model_service.builders.base import ReplicaSettings
    from model_service.builders.input_access import InputAccessFactory
    from model_service.providers.empaia.credentials import OutgoingCredentials
    from model_service.providers.empaia.registration import Marketplace
    from model_service.providers.empaia.triggers import PreprocessingTriggers


# The model files' block of the earlier EMPAIA offering; still accepted so old
# files load, but analyses' `expose.empaia` makes the apps now (transitional).
CATALOG_BLOCK = "empaia"

logger = logging.getLogger(__name__)


class EmpaiaProvider(Provider):
    """Runs EMPAIA jobs and registers the deployed analyses' EMPAIA apps."""

    catalog_block = CATALOG_BLOCK

    def api(
        self, catalog: ModelCatalogView, settings: SchedulerSettings
    ) -> ApiContribution:
        """Contribute the EMPAIA scheduler adapter, JES routes and app registration.

        Apps come from the deployed analyses with `expose.empaia`, as the
        catalog is when the API starts; admission reads the catalog in use.
        Registration runs in the background, and only when a marketplace is
        configured, so the API serves while the marketplace is down; it
        aligns the preprocessing triggers too when a Medical Data Service is
        configured. Models' `empaia` blocks are no longer read.

        Args:
            catalog: The deployed served models and analyses.
            settings: Unused; EMPAIA reads its own `EMPAIA_*` settings.

        Returns:
            the API contribution.

        Raises:
            pydantic.ValidationError: If the `EMPAIA_*` settings are invalid.
            jsonschema.ValidationError: If a generated EAD fails the EAD v3 schema.
        """
        from model_service.providers.empaia.ead.apps import analysis_apps
        from model_service.providers.empaia.jobs.adapter import EmpaiaProviderAdapter
        from model_service.providers.empaia.jobs.routes import create_router
        from model_service.providers.empaia.registration import register_until_done

        del settings
        empaia = EmpaiaSettings()
        if blocks := sorted(catalog.provider_block(CATALOG_BLOCK)):
            logger.warning(
                "model files' empaia blocks are no longer read (%s); an analysis's "
                "expose.empaia makes its app",
                ", ".join(blocks),
            )
        apps = analysis_apps(catalog.resolved)
        credentials = _credentials(empaia)
        marketplace = _marketplace(empaia, credentials)
        triggers = _triggers(empaia, credentials)

        async def register() -> None:
            """Register the apps until done, when a marketplace is configured.

            It runs with no apps too, so triggers of removed apps go.
            """
            if marketplace is not None:
                await register_until_done(marketplace, apps, triggers)

        return ApiContribution(
            adapter=EmpaiaProviderAdapter(
                http_gateway_factory(empaia),
                allowed_api_urls=empaia.api_urls,
                workspace_root=empaia.workspace_root,
                catalog=lambda: catalog.resolved,
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
