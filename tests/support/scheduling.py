"""Scheduler doubles: a scripted scheduler for route tests, a provider and a clock."""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_service.builders.contracts import (
    InputAccessContext,
    Invocation,
    SpatialScoresModel,
    UnknownModelError,
)
from model_service.scheduler.contracts import (
    ExecutionStatus,
    ExecutionView,
    ScheduleRequest,
    ScheduleResult,
    UnknownExecutionError,
)
from model_service.scheduler.domain import PreparedInvocation, SubmissionIdentity


if TYPE_CHECKING:
    from datetime import datetime, timedelta

    from model_service.builders.contracts import JsonValue
    from model_service.scheduler.providers import BeforeMutation, ProviderFinishOutcome


PROSTATE = SpatialScoresModel(
    id="prostate-classifier-1",
    tile_size=512,
    output_tile_size=1,
    n_channels=1,
    mpp=0.468,
)


def view(status: ExecutionStatus, **fields: object) -> ExecutionView:
    return ExecutionView(
        id="execution-1",
        builder_id="heatmap",
        model_id=PROSTATE.id,
        status=status,
        **fields,  # type: ignore[arg-type]
    )


class ScriptedScheduler:
    """Admits requests and answers status and cancellation from a fixed view."""

    def __init__(self, current: ExecutionView, *, created: bool = True) -> None:
        self.current = current
        self.created = created
        self.requests: list[ScheduleRequest] = []
        self.unknown_models: set[str] = set()
        self.cancelled: list[str] = []

    async def check_ready(self) -> None:
        return None

    async def schedule(self, request: ScheduleRequest) -> ScheduleResult:
        if request.model_id in self.unknown_models:
            raise UnknownModelError(request.model_id)
        self.requests.append(request)
        return ScheduleResult(self.current, created=self.created)

    async def get(self, execution_id: str) -> ExecutionView:
        if execution_id != self.current.id:
            raise UnknownExecutionError(execution_id)
        return self.current

    async def cancel(self, execution_id: str) -> ExecutionView:
        self.cancelled.append(execution_id)
        return await self.get(execution_id)


class StandaloneProvider:
    """Prepare minimal scheduler invocations without shared-file checks so tests can isolate orchestration."""

    name = "standalone"

    def identify(self, request: ScheduleRequest) -> SubmissionIdentity:
        """Use the test replay key with a fixed fingerprint to isolate scheduler replay behavior."""
        assert request.idempotency_key is not None
        return SubmissionIdentity(
            provider=self.name,
            key=request.idempotency_key,
            fingerprint="same-request",
        )

    async def prepare(
        self,
        request: ScheduleRequest,
        deadline_at: datetime,
    ) -> PreparedInvocation:
        """Create an empty shared-storage invocation with the scheduler-supplied deadline."""
        return PreparedInvocation(
            invocation=Invocation(
                inputs={},
                resolved_configuration={},
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
        _protected_context: dict[str, JsonValue],
        _outcome: ProviderFinishOutcome,
        *,
        before_mutation: BeforeMutation,
    ) -> None:
        """Complete without external writes or mutation guarding for standalone orchestration tests."""
        del before_mutation


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
