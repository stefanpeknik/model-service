"""The provider contract: what a provider is and every method it writes.

A provider is a `Provider` (here) listed in `providers/registry.py`. the API
and the builder ingress compose the providers there and name none of them.
A provider writes:

- `Provider.api` and `Provider.input_access`: what it adds to each process,
  plus `catalog_block`, the block of the model files it reads;
- a `ProviderAdapter` (here): `identify`, `prepare`
  and `finish`, admitting a job and finishing it;
- an `InputAccessFactory` (builders/input_access.py) whose `create` returns an
  `InputAccess`: `input_path`, `output_path`, `is_visible_artifact`, `aclose`.

The input-access interface lives with the builders that use it. Each provider imports
what only one process needs inside the method for it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Coroutine, Mapping
    from datetime import datetime

    from fastapi import APIRouter

    from model_service.builders.base import ReplicaSettings
    from model_service.builders.input_access import InputAccessFactory
    from model_service.model_catalog import ModelCatalog
    from model_service.scheduler.protocols import Capacity, Scheduler
    from model_service.scheduler.public import (
        ExecutionError,
        ProviderName,
        ScheduleRequest,
    )
    from model_service.scheduler.settings import SchedulerSettings
    from model_service.shared.invocation import InputAccessKind, Invocation
    from model_service.shared.json_values import JsonValue
    from model_service.shared.result import BuilderResult


type ProviderFinishOutcome = ProviderSuccess | ProviderFailure | ProviderCancellation
type BeforeMutation = Callable[[], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class SubmissionIdentity:
    """Scope a replay key to its provider and distinguish identical requests from conflicting ones."""

    provider: ProviderName
    key: str
    fingerprint: str


class Priority(IntEnum):
    """Order in which due work is claimed; higher first, within each model's capacity.

    Internal for now: providers set it, the public API does not take it.
    """

    LOW = 0
    NORMAL = 1
    HIGH = 2


@dataclass(frozen=True, slots=True)
class PreparedInvocation:
    """Carry a validated invocation, private provider state, and any provider-imposed deadline limits."""

    invocation: Invocation
    protected_context: dict[str, JsonValue]
    deadline_cap: datetime | None = None
    terminal_report_deadline_at: datetime | None = None
    priority: int = Priority.NORMAL


@dataclass(frozen=True, slots=True)
class ProviderSuccess:
    """Give provider finish the durable builder result to publish."""

    result: BuilderResult
    # The builder that produced it, whose declarations say what each output is.
    builder_id: str


@dataclass(frozen=True, slots=True)
class ProviderFailure:
    """Give provider finish the safe terminal processing error to report."""

    error: ExecutionError


@dataclass(frozen=True, slots=True)
class ProviderCancellation:
    """Ask provider finish to report the execution's cancellation."""


class ProviderFinishFailureError(RuntimeError):
    """Carry a safe publication error, retry permission, and whether a remote mutation may have happened."""

    def __init__(
        self,
        *,
        code: str,
        safe_message: str,
        retryable: bool,
        outcome_unknown: bool = False,
    ) -> None:
        """Capture finish failure metadata without exposing the underlying provider exception to callers.

        Args:
            code: Stable error code to record.
            safe_message: Message safe to show to callers.
            retryable: Whether the worker may retry the finish.
            outcome_unknown: Whether a remote mutation may have happened before the failure.
        """
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message
        self.retryable = retryable
        self.outcome_unknown = outcome_unknown


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
    ) -> None:
        """Publish or report a stored outcome to the provider.

        Awaits before_mutation before the first external change. Scheduling and
        retries remain worker-owned.

        Args:
            protected_context: Provider context stored at admission.
            outcome: Success, failure or cancellation to report.
            before_mutation: Callback to await before the first external change.

        Raises:
            ProviderFinishFailureError: Publication failed; carries retry permission and whether the outcome is unknown.
        """
        ...


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
