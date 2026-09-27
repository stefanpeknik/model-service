from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from dataclasses import replace
from typing import TYPE_CHECKING

from execution_scheduler.active_work import ActiveWorkRegistry
from execution_scheduler.contracts import (
    ExecutionError,
    ExecutionErrorCode,
)
from execution_scheduler.dispatch import (
    ConfirmedDispatchRejectionError,
    RemoteWorkflowFailureError,
    UnknownDispatchOutcomeError,
)
from execution_scheduler.domain import (
    ProcessingFailed,
    ProcessingRejected,
    ProcessingSucceeded,
    ProviderFinishFailed,
    ProviderFinishSucceeded,
    RenewDirective,
    StaleWorkClaimError,
    WorkClaim,
    WorkKind,
)
from execution_scheduler.providers import (
    ProviderCancellation,
    ProviderFailure,
    ProviderFinishFailureError,
    ProviderFinishOutcome,
    ProviderSuccess,
)


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import timedelta

    from execution_scheduler.dispatch import WorkflowDispatch
    from execution_scheduler.domain import Settlement
    from execution_scheduler.providers import ProviderAdapter
    from execution_scheduler.repository import ExecutionRepository
    from workflow_execution.contracts import WorkflowInvocation


class ExecutionWorker:
    def __init__(
        self,
        *,
        repository: ExecutionRepository,
        providers: Mapping[str, ProviderAdapter],
        dispatch: WorkflowDispatch,
        owner: str,
        lease_duration: timedelta,
        active_work: ActiveWorkRegistry | None = None,
    ) -> None:
        self._repository = repository
        self._providers = dict(providers)
        self._dispatch = dispatch
        self._owner = owner
        self._lease_duration = lease_duration
        self._active_work = active_work or ActiveWorkRegistry()

    async def run(self, stop: asyncio.Event, *, poll_interval: float) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        maintenance_task = asyncio.create_task(
            self._maintain(stop, poll_interval=poll_interval)
        )
        try:
            await self._run_claims(stop, poll_interval=poll_interval)
        finally:
            maintenance_task.cancel()
            await asyncio.gather(maintenance_task, return_exceptions=True)

    async def _maintain(self, stop: asyncio.Event, *, poll_interval: float) -> None:
        while not stop.is_set():
            try:
                await self._repository.maintain()
            except Exception as error:  # noqa: BLE001 - Keep maintenance alive after adapter failures.
                logger.error("execution maintenance failed: %s", type(error).__name__)
            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=poll_interval)

    async def _run_claims(self, stop: asyncio.Event, *, poll_interval: float) -> None:
        while not stop.is_set():
            try:
                worked = await self.run_once()
            except Exception as error:  # noqa: BLE001 - One failed claim must not stop the worker.
                logger.error(
                    "execution worker pass failed: %s",
                    type(error).__name__,
                )
                worked = False
            if worked:
                continue
            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=poll_interval)

    async def run_once(self) -> bool:
        claim = await self._repository.claim_due(
            self._owner,
            self._lease_duration,
        )
        if claim is None:
            return False
        task = asyncio.create_task(self._execute(claim))
        self._active_work.register(
            claim.execution_id,
            claim.fencing_token,
            task,
        )
        renewal_task = asyncio.create_task(self._renew_while_running(claim, task))
        try:
            done, _pending = await asyncio.wait(
                {task, renewal_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if renewal_task in done:
                directive = await renewal_task
                if directive is not RenewDirective.CONTINUE:
                    return True
            settlement = await task
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
            return True
        except Exception as error:
            logger.error(
                "execution claim failed: id=%s kind=%s attempt=%s token=%s error=%s",
                claim.execution_id,
                claim.kind,
                claim.attempt,
                claim.fencing_token,
                type(error).__name__,
            )
            raise
        finally:
            for pending_task in (task, renewal_task):
                if not pending_task.done():
                    pending_task.cancel()
            await asyncio.gather(task, renewal_task, return_exceptions=True)
            self._active_work.unregister(claim.execution_id, claim.fencing_token)
        try:
            await self._repository.settle(
                claim.execution_id,
                claim.fencing_token,
                settlement,
            )
        except StaleWorkClaimError:
            return True
        return True

    async def _renew_while_running(
        self,
        claim: WorkClaim,
        task: asyncio.Task[Settlement],
    ) -> RenewDirective:
        interval = max(self._lease_duration.total_seconds() / 3, 0.001)
        while not task.done():
            await asyncio.sleep(interval)
            if task.done():
                return RenewDirective.CONTINUE
            async with asyncio.timeout(interval):
                directive = await self._repository.renew(
                    claim.execution_id,
                    claim.fencing_token,
                    self._lease_duration,
                )
            if directive is not RenewDirective.CONTINUE:
                return directive
        return RenewDirective.CONTINUE

    async def _execute(self, claim: WorkClaim) -> Settlement:
        if claim.kind is WorkKind.PROCESSING:
            try:
                result = await self._dispatch.run(
                    claim.workflow_id,
                    claim.fencing_token,
                    _invocation_with_protected_context(claim),
                )
                return ProcessingSucceeded(result)
            except ConfirmedDispatchRejectionError as error:
                return ProcessingRejected(error.retry_after)
            except UnknownDispatchOutcomeError:
                return ProcessingFailed(
                    error=ExecutionError(
                        code=ExecutionErrorCode.WORKFLOW_OUTCOME_UNKNOWN,
                        message="workflow dispatch outcome is unknown",
                    ),
                    retryable=True,
                )
            except RemoteWorkflowFailureError as error:
                return ProcessingFailed(
                    error=ExecutionError(error.code, error.safe_message),
                    retryable=error.retryable,
                )
        try:
            mutation_started = False

            async def before_mutation() -> None:
                nonlocal mutation_started
                if mutation_started:
                    return
                await self._repository.mark_finish_mutation_started(
                    claim.execution_id,
                    claim.fencing_token,
                )
                mutation_started = True

            await self._providers[claim.provider].finish(
                claim.protected_context,
                _finish_outcome(claim),
                before_mutation=before_mutation,
            )
            return ProviderFinishSucceeded()
        except ProviderFinishFailureError as error:
            return ProviderFinishFailed(
                error=ExecutionError(error.code, error.safe_message),
                retryable=error.retryable,
                outcome_unknown=error.outcome_unknown,
            )


def _invocation_with_protected_context(claim: WorkClaim) -> WorkflowInvocation:
    access = replace(
        claim.invocation.input_access,
        state={
            **claim.invocation.input_access.state,
            **claim.protected_context,
            "processing_attempt_id": claim.fencing_token,
        },
    )
    return replace(claim.invocation, input_access=access)


def _finish_outcome(claim: WorkClaim) -> ProviderFinishOutcome:
    if claim.kind is WorkKind.PROVIDER_FINISH:
        if claim.result is None:
            raise RuntimeError("provider finish requires a durable result")
        return ProviderSuccess(claim.result)
    if claim.kind is WorkKind.PROVIDER_FAILURE_REPORT:
        if claim.error is None:
            raise RuntimeError("provider failure report requires a durable error")
        return ProviderFailure(claim.error)
    if claim.kind is WorkKind.PROVIDER_CANCELLATION_REPORT:
        return ProviderCancellation()
    raise RuntimeError("unsupported provider work kind")
