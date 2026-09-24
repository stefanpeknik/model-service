from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from datetime import datetime

    from execution_scheduler.contracts import ExecutionError, ScheduleRequest
    from execution_scheduler.domain import PreparedWorkflow, SubmissionIdentity
    from workflow_execution.contracts import JsonValue, WorkflowResult


@dataclass(frozen=True, slots=True)
class ProviderSuccess:
    result: WorkflowResult


@dataclass(frozen=True, slots=True)
class ProviderFailure:
    error: ExecutionError


@dataclass(frozen=True, slots=True)
class ProviderCancellation:
    pass


type ProviderFinishOutcome = ProviderSuccess | ProviderFailure | ProviderCancellation
type BeforeMutation = Callable[[], Awaitable[None]]


class ProviderFinishFailureError(RuntimeError):
    def __init__(
        self,
        *,
        code: str,
        safe_message: str,
        retryable: bool,
        outcome_unknown: bool = False,
    ) -> None:
        super().__init__(safe_message)
        self.code = code
        self.safe_message = safe_message
        self.retryable = retryable
        self.outcome_unknown = outcome_unknown


class ProviderAdapter(Protocol):
    name: str

    def identify(self, request: ScheduleRequest) -> SubmissionIdentity: ...

    async def prepare(
        self,
        request: ScheduleRequest,
        deadline_at: datetime,
    ) -> PreparedWorkflow: ...

    async def finish(
        self,
        protected_context: dict[str, JsonValue],
        outcome: ProviderFinishOutcome,
        *,
        before_mutation: BeforeMutation,
    ) -> None: ...
