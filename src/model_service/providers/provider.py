"""The provider contract: what a provider is and every method it writes.

A provider is a `Provider` (here) listed in `providers/registry.py`. The facade
and the builder ingress compose the providers there and name none of them.
A provider writes:

- `Provider.facade` and `Provider.input_access`: what it adds to each process,
  plus `catalog_block`, the block of the model files it reads;
- a `ProviderAdapter` (scheduler/provider_adapter.py): `identify`, `prepare`
  and `finish`, admitting a job and finishing it;
- an `InputAccessFactory` (builders/input_access.py) whose `create` returns an
  `InputAccess`: `input_path`, `output_path`, `is_visible_artifact`, `aclose`.

Those interfaces live with the packages that call them. Each provider imports
what only one process needs inside the method for it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Mapping

    from fastapi import APIRouter

    from model_service.builder_service.settings import ReplicaSettings
    from model_service.builders.input_access import InputAccessFactory
    from model_service.catalog import ModelCatalog
    from model_service.scheduler.api_interface import Capacity, Scheduler
    from model_service.scheduler.provider_adapter import ProviderAdapter
    from model_service.scheduler.settings import SchedulerSettings
    from model_service.wire.invocation import InputAccessKind


type RouteFactory = Callable[[Scheduler, Capacity], APIRouter]


@dataclass(frozen=True, slots=True)
class FacadeContribution:
    adapter: ProviderAdapter
    routes: RouteFactory | None = None
    # Runs while the facade serves and is cancelled when it stops.
    background: Callable[[], Coroutine[None, None, None]] | None = None


class Provider(ABC):
    # The model files' top-level block it reads, if any.
    catalog_block: str | None

    @abstractmethod
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

    @abstractmethod
    def input_access(
        self, settings: ReplicaSettings
    ) -> Mapping[InputAccessKind, InputAccessFactory]:
        """Build the adapters builder replicas use for the inputs this provider hands out.

        Args:
            settings: The builder replica's settings.

        Returns:
            The access adapters by the input access kind they serve.
        """
        ...
