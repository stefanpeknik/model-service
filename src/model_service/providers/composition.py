"""What each provider adds to the facade and to builder replicas.

The facade and the builder ingress compose the providers in `PROVIDERS` and
name none of them. A provider contributes its scheduler adapter, its routes and
background work to the facade, reads its block of the model files, and gives
builder replicas the access adapters for the inputs it hands out. Each
provider imports what only one process needs inside the method for it.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Mapping

    from fastapi import APIRouter

    from model_service.builders.application import AccessAdapter
    from model_service.builders.contracts import InputAccessKind
    from model_service.builders.settings import WorkerSettings
    from model_service.facade.catalog import ModelCatalog
    from model_service.facade.routes.common import Capacity, Scheduler
    from model_service.facade.settings import SchedulerSettings
    from model_service.scheduler.providers import ProviderAdapter


type RouteFactory = Callable[[Scheduler, Capacity], APIRouter]


@dataclass(frozen=True, slots=True)
class FacadeContribution:
    adapter: ProviderAdapter
    routes: RouteFactory | None = None
    # Runs while the facade serves and is cancelled when it stops.
    background: Callable[[], Coroutine[None, None, None]] | None = None


class Provider(Protocol):
    # The model files' top-level block it reads, if any.
    catalog_block: str | None

    def facade(
        self, catalog: ModelCatalog, settings: SchedulerSettings
    ) -> FacadeContribution:
        """Build what this provider adds to the facade.

        Args:
            catalog: The deployed models, with each model file's provider blocks.
            settings: The facade's scheduler settings.

        Returns:
            The provider's scheduler adapter, and its routes and background work if any.
        """
        ...

    def input_access(
        self, settings: WorkerSettings
    ) -> Mapping[InputAccessKind, AccessAdapter]:
        """Build the adapters builder replicas use for the inputs this provider hands out.

        Args:
            settings: The builder replica's settings.

        Returns:
            The access adapters by the input access kind they serve.
        """
        ...


PROVIDERS = (
    "model_service.providers.standalone:PROVIDER",
    "model_service.providers.empaia.composition:PROVIDER",
)


def load_providers(names: tuple[str, ...] = PROVIDERS) -> list[Provider]:
    """Import the providers named as `module:attribute`.

    Args:
        names: Provider references, in the order to compose them.

    Returns:
        The providers, in the order of `names`.

    Raises:
        ImportError: If a named module cannot be imported.
        AttributeError: If a module lacks the named attribute.
    """
    providers = []
    for name in names:
        module, _, attribute = name.partition(":")
        providers.append(getattr(importlib.import_module(module), attribute))
    return providers
