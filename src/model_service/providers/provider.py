"""The provider contract: what a provider is and every method it writes.

A provider is a `Provider` (here) listed in `providers/registry.py`. The control plane
process and the pipeline runner compose the providers there and name none
of them. A provider writes:

- `Provider.control_plane` and `Provider.input_access`: what it adds to each process;
- a `ProviderAdapter` (here): `identify`, `prepare` and `finish`, admitting a
  job and finishing it, with the types they take and return in
  control_plane/scheduler/public.py;
- an `InputAccessFactory` (providers/input_access.py) whose `create` returns an
  `InputAccess`: `input_path`, `output_path`, `aclose`.

Each provider imports what only one process needs inside the method for it:
the pipeline runner loads provider modules, so those import no api/ code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable, Coroutine, Mapping
    from datetime import datetime

    from fastapi import APIRouter

    from model_service.catalog import Catalog
    from model_service.contracts.invocation import InputAccessKind
    from model_service.contracts.json_values import JsonValue
    from model_service.contracts.result import RunResult
    from model_service.control_plane.scheduler.protocols import Scheduler
    from model_service.control_plane.scheduler.public import (
        BeforeMutation,
        PipelineScheduleRequest,
        PreparedInvocation,
        ProviderFinishOutcome,
        ProviderName,
        SubmissionIdentity,
    )
    from model_service.control_plane.scheduler.settings import SchedulerSettings
    from model_service.pipeline_runner.settings import ReplicaSettings
    from model_service.providers.input_access import InputAccessFactory


class ProviderAdapter(ABC):
    """Translate one integration's replay identity, prepared invocation, and terminal reporting behavior."""

    name: ProviderName

    @abstractmethod
    def identify(self, request: PipelineScheduleRequest) -> SubmissionIdentity:
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
        request: PipelineScheduleRequest,
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
    ) -> RunResult | None:
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
class ControlPlaneContribution:
    """What a provider adds to the control plane.

    Attributes:
        adapter: Its scheduler adapter.
        routes: Its routes, if any.
        background: Work that runs while the control plane serves, cancelled when it
            stops.
    """

    adapter: ProviderAdapter
    routes: RouteFactory | None = None
    # Runs while the control plane serves and is cancelled when it stops.
    background: Callable[[], Coroutine[None, None, None]] | None = None


class Provider(ABC):
    """A source of jobs: how they are admitted, read and finished."""

    @abstractmethod
    def control_plane(
        self, catalog: Callable[[], Catalog], settings: SchedulerSettings
    ) -> ControlPlaneContribution:
        """Build what this provider adds to the control plane.

        Args:
            catalog: Gives the deployed models and pipelines in use.
            settings: the control plane's scheduler settings.

        Returns:
            The provider's scheduler adapter, and its routes and background work if any.
        """
        ...

    @abstractmethod
    def input_access(
        self, settings: ReplicaSettings
    ) -> Mapping[InputAccessKind, InputAccessFactory]:
        """Build the adapters the pipeline runner uses for the inputs this provider hands out.

        Args:
            settings: The runner replica's settings.

        Returns:
            The access adapters by the input access kind they serve.
        """
        ...
