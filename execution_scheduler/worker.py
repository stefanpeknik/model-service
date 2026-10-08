"""Run leased processing and provider attempts with independent maintenance and renewal.

Each worker handles one claim at a time. Computation results are settled before provider
finish is claimed, allowing publication retries without rerunning the workflow.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import TYPE_CHECKING

from execution_scheduler.contracts import (
    ExecutionError,
    ExecutionErrorCode,
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
from workflow_execution.contracts import WorkflowExecutionContext
from workflow_execution.dispatch import (
    ConfirmedDispatchRejectionError,
    RemoteWorkflowFailureError,
    UnknownDispatchOutcomeError,
)


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import timedelta

    from execution_scheduler.active_work import ActiveWorkRegistry
    from execution_scheduler.domain import Settlement
    from execution_scheduler.providers import ProviderAdapter
    from execution_scheduler.repository import ExecutionRepository
    from workflow_execution.dispatch import WorkflowDispatch


class ExecutionWorker:
    """Execute due claims while monitoring durable ownership and recording safe outcomes."""

    def __init__(
        self,
        *,
        repository: ExecutionRepository,
        providers: Mapping[str, ProviderAdapter],
        dispatch: WorkflowDispatch,
        owner: str,
        lease_duration: timedelta,
        active_work: ActiveWorkRegistry,
    ) -> None:
        """Store caller-owned adapters, worker identity, lease timing, and the shared local task registry.

        Args:
            repository: Atomic execution storage boundary; the adapter controls
                durability.
            providers: Provider name to integration adapter mapping enabled in this
                scheduler.
            dispatch: Workflow transport used to invoke the selected Ray Serve
                application.
            owner: Unique worker identity recorded on the durable lease.
            lease_duration: How long a claimed attempt retains ownership without
                renewal.
            active_work: Process-local registry shared with admission/cancellation
                orchestration.
        """
        self._repository = repository
        self._providers = dict(providers)
        self._dispatch = dispatch
        self._owner = owner
        self._lease_duration = lease_duration
        self._active_work = active_work

    async def run(self, stop: asyncio.Event, *, poll_interval: float) -> None:
        """Run maintenance alongside the claim loop and join maintenance when the worker stops.

        Args:
            stop: Event signalled when the owning worker/resolver lifetime should end.
            poll_interval: Positive interval between maintenance passes and empty/failed
                claim polls.

        Returns:
            None: No return value.

        Raises:
            ValueError: poll_interval is nonpositive.
            asyncio.CancelledError: The owning worker task is cancelled; maintenance is
                joined before propagation.
        """
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
        """Poll deadline and lease maintenance, logging adapter failures without stopping subsequent passes.

        Args:
            stop: Event signalled when the owning worker/resolver lifetime should end.
            poll_interval: Positive interval between maintenance passes and empty/failed
                claim polls.

        Returns:
            None: No return value.
        """
        while not stop.is_set():
            try:
                await self._repository.maintain()
            except Exception as error:
                logger.error("execution maintenance failed: %s", type(error).__name__)
            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=poll_interval)

    async def _run_claims(self, stop: asyncio.Event, *, poll_interval: float) -> None:
        """Process claims sequentially and wait between empty or failed passes until shutdown.

        Args:
            stop: Event signalled when the owning worker/resolver lifetime should end.
            poll_interval: Positive interval between maintenance passes and empty/failed
                claim polls.

        Returns:
            None: No return value.
        """
        while not stop.is_set():
            try:
                worked = await self.run_once()
            except Exception as error:
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
        """Claim one attempt, register its task, renew its lease, and settle only a valid completed outcome.

        Lost ownership stops local work. Shutdown cancellation propagates; cancellation
        of the attempt alone ends this pass. Cleanup joins both tasks and removes only
        this attempt's registry entry.

        Returns:
            bool: True if a claim was acquired, even if cancelled or stale; False when
                no eligible work was found.

        Raises:
            ExecutionRepositoryUnavailableError: Claiming, renewing, or settling work
                cannot access durable storage.
            asyncio.CancelledError: The owning worker lifetime is cancelled; local claim
                tasks are joined before propagation.
        """
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
        """Renew about every third of a lease and bound each renewal call by that interval. Return a stop directive if the claim is invalid.

        Args:
            claim: Leased attempt with current work kind, token, invocation, and stored
                outcome.
            task: Local computation or cleanup task to cancel, await, or release.

        Returns:
            RenewDirective: Instruction to continue or stop the current attempt.

        Raises:
            ExecutionRepositoryUnavailableError: Storage cannot confirm continued
                ownership.
            TimeoutError: A renewal call exceeds one third of the lease duration.
            UnknownExecutionError: The claimed execution no longer exists.
        """
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
        """Dispatch processing or call the provider's finish hook and translate known failures into settlement values.

        Args:
            claim: Leased attempt with current work kind, token, invocation, and stored
                outcome.

        Returns:
            Settlement: Typed processing or provider outcome to settle through the
                repository.

        Raises:
            StaleWorkClaimError: A provider mutation guard rejects a stale lease or
                deadline.
            asyncio.CancelledError: The local attempt is cancelled after shutdown,
                cancellation, or loss of ownership.
        """
        if claim.kind is WorkKind.PROCESSING:
            try:
                result = await self._dispatch.run(
                    claim.workflow_id,
                    WorkflowExecutionContext(
                        processing_attempt_id=claim.fencing_token,
                        protected_context=claim.protected_context,
                    ),
                    claim.invocation,
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
                """Persist the provider mutation boundary once, immediately before its first external write.

                Returns:
                    None: No return value.

                Raises:
                    ExecutionRepositoryUnavailableError: Storage cannot persist the
                        external-mutation marker.
                    StaleWorkClaimError: The claim is no longer authorized to publish
                        its provider outcome.
                    UnknownExecutionError: The claimed execution no longer exists.
                """
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


def _finish_outcome(claim: WorkClaim) -> ProviderFinishOutcome:
    """Convert a provider claim into success, failure, or cancellation, requiring the corresponding stored result or error.

    Args:
        claim: Leased attempt with current work kind, token, invocation, and stored
            outcome.

    Returns:
        ProviderFinishOutcome: Success, failure, or cancellation derived from the stored
            provider claim.

    Raises:
        RuntimeError: Publication lacks a stored result, failure reporting lacks a
            stored error, or the claim is not provider work.
    """
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
