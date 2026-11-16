"""Standalone provider: shared-storage inputs and outputs, no external job lifecycle.

Builder replicas load this module through the provider registry, so it imports
no API code; the API side (admission, finish) is adapter.py, imported lazily.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_service.contracts.invocation import SHARED_STORAGE
from model_service.providers.provider import Provider
from model_service.providers.standalone.shared_storage import SharedStorageAccessFactory


if TYPE_CHECKING:
    from collections.abc import Mapping

    from model_service.api.model_catalog import ModelCatalog
    from model_service.builders.base import ReplicaSettings
    from model_service.builders.input_access import InputAccessFactory
    from model_service.providers.provider import ApiContribution
    from model_service.scheduler.settings import SchedulerSettings


# The provider name of requests made through the public builder routes.
STANDALONE_PROVIDER = "standalone"


class StandaloneProvider(Provider):
    """Shared-storage jobs, through the public builder routes the API serves."""

    catalog_block = None

    def api(
        self, catalog: ModelCatalog, settings: SchedulerSettings
    ) -> ApiContribution:
        """Contribute the standalone scheduler adapter, with no routes or background work.

        the API's public builder routes submit these jobs.

        Args:
            catalog: Unused.
            settings: Supplies the allowed shared-storage roots.

        Returns:
            the API contribution.
        """
        from model_service.providers.provider import ApiContribution
        from model_service.providers.standalone.adapter import (
            StandaloneProviderAdapter,
        )

        del catalog
        return ApiContribution(
            StandaloneProviderAdapter(
                SharedStorageAccessFactory(settings.shared_storage_roots)
            )
        )

    def input_access(
        self, settings: ReplicaSettings
    ) -> Mapping[str, InputAccessFactory]:
        """Give builder replicas shared-storage access below the allowed roots.

        Args:
            settings: Supplies the allowed shared-storage roots.

        Returns:
            The shared-storage adapter, keyed by `SHARED_STORAGE`.
        """
        return {
            SHARED_STORAGE: SharedStorageAccessFactory(settings.shared_storage_roots)
        }


PROVIDER = StandaloneProvider()
