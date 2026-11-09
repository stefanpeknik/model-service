"""What each provider adds to the facade and to builder replicas.

The facade and the builder ingress compose the providers in
`providers/installed.py` and
name none of them. A provider contributes its scheduler adapter and its routes
to the facade, and gives builder replicas the access adapters for the inputs it hands out. Each
provider imports what only one process needs inside the method for it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from fastapi import APIRouter

    from model_service.builders.access import AccessAdapter
    from model_service.builders.contracts import InputAccessKind
    from model_service.builders.settings import WorkerSettings
    from model_service.facade.ports import Capacity, Scheduler
    from model_service.facade.settings import SchedulerSettings
    from model_service.scheduler.providers import ProviderAdapter


type RouteFactory = Callable[[Scheduler, Capacity], APIRouter]


@dataclass(frozen=True, slots=True)
class FacadeContribution:
    adapter: ProviderAdapter
    routes: RouteFactory | None = None


class Provider(ABC):
    @abstractmethod
    def facade(self, settings: SchedulerSettings) -> FacadeContribution:
        """Build what this provider adds to the facade.

        Args:
            settings: The facade's scheduler settings.

        Returns:
            The provider's scheduler adapter, and its routes if any.
        """
        ...

    @abstractmethod
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
