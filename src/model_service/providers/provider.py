"""The provider contract: how the scheduler admits and finishes a provider's jobs.

A provider writes a `ProviderAdapter` (here): `identify`, `prepare` and
`finish`, admitting a job and finishing it, with the types they take and return
in scheduler/public.py. Its inputs reach builders through an
`InputAccessFactory` (builders/input_access.py).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from datetime import datetime

    from model_service.contracts.json_values import JsonValue
    from model_service.contracts.result import BuilderResult
    from model_service.scheduler.public import (
        BeforeMutation,
        PreparedInvocation,
        ProviderFinishOutcome,
        ProviderName,
        ScheduleRequest,
        SubmissionIdentity,
    )


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
