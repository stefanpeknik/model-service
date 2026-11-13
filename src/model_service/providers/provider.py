"""The provider contract: what a provider is and every method it writes.

A provider is a `Provider` (here) listed in `providers/registry.py`. The API
process and the builder deployment compose the providers there and name none
of them. A provider writes:

- `Provider.api` and `Provider.input_access`: what it adds to each process,
  plus `catalog_block`, the block of the model files it reads;
- a `ProviderAdapter` (here): `identify`, `prepare` and `finish`, admitting a
  job and finishing it, with the types they take and return in
  scheduler/public.py;
- an `InputAccessFactory` (builders/input_access.py) whose `create` returns an
  `InputAccess`: `input_path`, `output_path`, `is_visible_artifact`, `aclose`.

Each provider imports what only one process needs inside the method for it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Mapping
    from datetime import datetime

    from fastapi import APIRouter

    from model_service.builders.base import ReplicaSettings
    from model_service.builders.input_access import InputAccessFactory
    from model_service.contracts.invocation import InputAccessKind
    from model_service.contracts.json_values import JsonValue
    from model_service.contracts.result import BuilderResult
    from model_service.deployed_models.catalog import ModelCatalog
    from model_service.scheduler.protocols import Scheduler
    from model_service.scheduler.public import (
        BeforeMutation,
        PreparedInvocation,
        ProviderFinishOutcome,
        ProviderName,
        ScheduleRequest,
        SubmissionIdentity,
    )
    from model_service.scheduler.settings import SchedulerSettings


class ProviderAdapter(ABC):
    """Translate one integration's replay identity, prepared invocation, and terminal reporting behavior."""

    name: ProviderName

    @abstractmethod
    def identify(self, request: ScheduleRequest) -> SubmissionIdentity:
        """Compute submission identity locally without I/O or external side effects.

        Args:
            request: Provider schedule request.

        Returns:
            The replay identity of the request.
        """
        ...

    @abstractmethod
    async def prepare(
        self,
        request: ScheduleRequest,
        deadline_at: datetime,
    ) -> PreparedInvocation:
        """Resolve and validate new inputs and configuration before admission.

        Args:
            request: Provider schedule request.
            deadline_at: Execution deadline the preparation must respect.

        Returns:
            The prepared invocation with protected context and deadline limits.
        """
        ...

    @abstractmethod
    async def finish(
        self,
        protected_context: dict[str, JsonValue],
        outcome: ProviderFinishOutcome,
        *,
        before_mutation: BeforeMutation,
    ) -> BuilderResult | None:
        """Publish or report a stored outcome to the provider.

        Awaits before_mutation before the first external change. Scheduling and
        retries remain worker-owned.

        Args:
            protected_context: Provider context stored at admission.
            outcome: Success, failure or cancellation to report.
            before_mutation: Callback to await before the first external change.

        Returns:
            A result to store in place of the processing result, e.g. with moved
            artifact paths, or None to keep it.

        Raises:
            ProviderFinishFailureError: Publication failed; carries retry permission and whether the outcome is unknown.
        """
        ...


type RouteFactory = Callable[[Scheduler], APIRouter]


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
