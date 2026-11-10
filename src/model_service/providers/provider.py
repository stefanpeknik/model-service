"""The provider contract: what a provider is and every method it writes.

A provider is a `Provider` (here) listed in `providers/registry.py`. the API
and the builder ingress compose the providers there and name none of them.
A provider writes:

- `Provider.api` and `Provider.input_access`: what it adds to each process,
  plus `catalog_block`, the block of the model files it reads;
- a `ProviderAdapter` (providers/adapter.py): `identify`, `prepare`
  and `finish`, admitting a job and finishing it;
- an `InputAccessFactory` (builders/input_access.py) whose `create` returns an
  `InputAccess`: `input_path`, `output_path`, `is_visible_artifact`, `aclose`.

The input-access interface lives with the builders that use it. Each provider imports
what only one process needs inside the method for it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Mapping

    from fastapi import APIRouter

    from model_service.builders.base import ReplicaSettings
    from model_service.builders.input_access import InputAccessFactory
    from model_service.model_catalog import ModelCatalog
    from model_service.providers.adapter import ProviderAdapter
    from model_service.scheduler.protocols import Capacity, Scheduler
    from model_service.scheduler.settings import SchedulerSettings
    from model_service.wire.invocation import InputAccessKind


type RouteFactory = Callable[[Scheduler, Capacity], APIRouter]


@dataclass(frozen=True, slots=True)
class ApiContribution:
    adapter: ProviderAdapter
    routes: RouteFactory | None = None
    # Runs while the API serves and is cancelled when it stops.
    background: Callable[[], Coroutine[None, None, None]] | None = None


class Provider(ABC):
    # The model files' top-level block it reads, if any.
    catalog_block: str | None

    @abstractmethod
    def api(
        self, catalog: ModelCatalog, settings: SchedulerSettings
    ) -> ApiContribution:
        """Build what this provider adds to the API.

        Args:
            catalog: The deployed models, with each model file's provider blocks.
            settings: the API's scheduler settings.

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
