"""Scheduler doubles shared by scheduler tests: a minimal provider and a clock."""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_service.api.scheduler.public import PreparedInvocation, SubmissionIdentity
from model_service.contracts.invocation import InputAccessContext, Invocation
from model_service.providers.provider import ProviderAdapter


if TYPE_CHECKING:
    from datetime import datetime, timedelta

    from model_service.api.scheduler.public import (
        AnyScheduleRequest,
        BeforeMutation,
        ProviderFinishOutcome,
    )
    from model_service.contracts.json_values import JsonValue


class StandaloneProvider(ProviderAdapter):
    """Prepare minimal scheduler invocations without shared-file checks so tests can isolate orchestration."""

    name = "standalone"

    def identify(self, request: AnyScheduleRequest) -> SubmissionIdentity:
        """Use the test replay key with a fixed fingerprint to isolate scheduler replay behavior."""
        assert request.idempotency_key is not None
        return SubmissionIdentity(
            provider=self.name,
            key=request.idempotency_key,
            fingerprint="same-request",
        )

    async def prepare(
        self,
        request: AnyScheduleRequest,
        deadline_at: datetime,
    ) -> PreparedInvocation:
        """Create an empty shared-storage invocation with the scheduler-supplied deadline."""
        return PreparedInvocation(
            invocation=Invocation(
                inputs={},
                configuration={},
                input_access=InputAccessContext(
                    kind="shared_storage",
                    state={"root": "/shared"},
                ),
                deadline_at=deadline_at,
            ),
            protected_context={},
        )

    async def finish(
        self,
        protected_context: dict[str, JsonValue],
        outcome: ProviderFinishOutcome,
        *,
        before_mutation: BeforeMutation,
    ) -> None:
        """Complete without external writes or mutation guarding for standalone orchestration tests."""
        del protected_context, outcome, before_mutation


class MutableClock:
    """Provide a manually advanced timestamp for scheduler lease, retry, and deadline tests."""

    def __init__(self, now: datetime) -> None:
        """Set the initial aware timestamp supplied by the test."""
        self.now = now

    def __call__(self) -> datetime:
        """Return the current test timestamp without advancing it."""
        return self.now

    def advance(self, duration: timedelta) -> None:
        """Advance simulated time by the supplied duration without sleeping."""
        self.now += duration
