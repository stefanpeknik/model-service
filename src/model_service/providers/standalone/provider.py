"""Standalone provider: shared-storage inputs and outputs, no external job lifecycle.

The pipeline runner loads this module through the provider registry, so it imports
no control-plane code; the control-plane side (admission, finish) is adapter.py, imported lazily.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_service.contracts.invocation import SHARED_STORAGE
from model_service.providers.provider import Provider
from model_service.providers.standalone.shared_storage import SharedStorageAccessFactory


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from model_service.catalog import Catalog
    from model_service.control_plane.scheduler.settings import SchedulerSettings
    from model_service.pipeline_runner.settings import ReplicaSettings
    from model_service.providers.input_access import InputAccessFactory
    from model_service.providers.provider import ControlPlaneContribution


# The provider name of requests made through the public pipeline routes.
STANDALONE_PROVIDER = "standalone"


class StandaloneProvider(Provider):
    """Shared-storage jobs, through the public pipeline routes the control plane serves."""

    def control_plane(
        self, catalog: Callable[[], Catalog], settings: SchedulerSettings
    ) -> ControlPlaneContribution:
        """Contribute the standalone scheduler adapter, with no routes or background work.

        The control plane's public pipeline routes submit these jobs.

        Args:
            catalog: Unused.
            settings: Supplies the allowed shared-storage roots.

        Returns:
            The control plane contribution.
        """
        from model_service.providers.provider import ControlPlaneContribution
        from model_service.providers.standalone.adapter import (
            StandaloneProviderAdapter,
        )

        del catalog
        return ControlPlaneContribution(
            StandaloneProviderAdapter(
                SharedStorageAccessFactory(settings.shared_storage_roots)
            )
        )

    def input_access(
        self, settings: ReplicaSettings
    ) -> Mapping[str, InputAccessFactory]:
        """Give the pipeline runner shared-storage access below the allowed roots.

        Args:
            settings: Supplies the allowed shared-storage roots.

        Returns:
            The shared-storage adapter, keyed by `SHARED_STORAGE`.
        """
        return {
            SHARED_STORAGE: SharedStorageAccessFactory(settings.shared_storage_roots)
        }


PROVIDER = StandaloneProvider()
