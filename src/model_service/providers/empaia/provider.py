"""The EMPAIA provider, composed for the API and for builder replicas.

It serves JES's remote execution routes and runs App Service jobs. Its settings
come from `EMPAIA_*` (`EmpaiaSettings`).
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


# The model files' block it reads: how a model is offered in EMPAIA.
CATALOG_BLOCK = "empaia"


class EmpaiaProvider(Provider):
    """Runs EMPAIA jobs and publishes their outputs."""

    catalog_block = CATALOG_BLOCK

    def api(
        self, catalog: ModelCatalog, settings: SchedulerSettings
    ) -> ApiContribution:
        """Contribute the EMPAIA scheduler adapter and the JES routes.

        Args:
            catalog: Unused until apps are registered.
            settings: Unused; EMPAIA reads its own `EMPAIA_*` settings.

        Returns:
            the API contribution.

        Raises:
            pydantic.ValidationError: If the `EMPAIA_*` settings are invalid.
        """
        from model_service.providers.empaia.jobs.adapter import EmpaiaProviderAdapter
        from model_service.providers.empaia.jobs.routes import create_router
        from model_service.providers.empaia.publishing.publishers import (
            default_publishers,
        )

        del catalog, settings
        empaia = EmpaiaSettings()
        return ApiContribution(
            adapter=EmpaiaProviderAdapter(
                http_gateway_factory(empaia),
                allowed_api_urls=empaia.api_urls,
                workspace_root=empaia.workspace_root,
                publishers=default_publishers(),
            ),
            routes=create_router,
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


PROVIDER = EmpaiaProvider()
