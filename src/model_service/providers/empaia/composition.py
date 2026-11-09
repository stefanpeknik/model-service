"""The EMPAIA provider, composed for the facade and for builder replicas.

It serves JES's remote execution routes and runs App Service jobs. Its
settings come from `EMPAIA_*` (`EmpaiaSettings`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_service.providers.composition import FacadeContribution, Provider
from model_service.providers.empaia.gateway import INPUT_ACCESS_KIND
from model_service.providers.empaia.http_gateway import http_gateway_factory
from model_service.providers.empaia.settings import EmpaiaSettings


if TYPE_CHECKING:
    from collections.abc import Mapping

    from model_service.builders.access import AccessAdapter
    from model_service.builders.settings import WorkerSettings
    from model_service.facade.settings import SchedulerSettings


class EmpaiaProvider(Provider):
    def facade(self, settings: SchedulerSettings) -> FacadeContribution:
        """Contribute the EMPAIA scheduler adapter and JES routes.

        Args:
            settings: Unused; EMPAIA reads its own `EMPAIA_*` settings.

        Returns:
            The facade contribution.

        Raises:
            pydantic.ValidationError: If the `EMPAIA_*` settings are invalid.
        """
        from model_service.providers.empaia.provider import EmpaiaProviderAdapter
        from model_service.providers.empaia.routes import create_router

        del settings
        empaia = EmpaiaSettings()
        return FacadeContribution(
            adapter=EmpaiaProviderAdapter(
                http_gateway_factory(empaia),
                allowed_api_urls=empaia.api_urls,
                workspace_root=empaia.workspace_root,
            ),
            routes=create_router,
        )

    def input_access(self, settings: WorkerSettings) -> Mapping[str, AccessAdapter]:
        """Give builder replicas access to EMPAIA job inputs through the App Service.

        Args:
            settings: Unused; EMPAIA reads its own `EMPAIA_*` settings.

        Returns:
            The EMPAIA input access adapter, keyed by `INPUT_ACCESS_KIND`.

        Raises:
            pydantic.ValidationError: If the `EMPAIA_*` settings are invalid.
        """
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


PROVIDER = EmpaiaProvider()
