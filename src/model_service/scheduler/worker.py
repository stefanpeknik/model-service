"""Run leased processing and provider attempts with independent maintenance and renewal.

Results are settled before provider finish is claimed, so a failed publication is
retried without rerunning the builder.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import TYPE_CHECKING

from model_service.contracts.invocation import ExecutionContext
from model_service.contracts.result import BuilderResult
from model_service.scheduler.builder_dispatch import (
    ConfirmedRejectionError,
    RemoteFailureError,
    UnknownOutcomeError,
)
from model_service.scheduler.internal_state import (
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
from model_service.scheduler.public import (
    ExecutionError,
    IncompatibleModelError,
    ProviderCancellation,
    ProviderFailure,
    ProviderFinishFailureError,
    ProviderFinishOutcome,
    ProviderSuccess,
    UnknownBuilderError,
    UnknownModelError,
)
from model_service.scheduler.status import ExecutionErrorCode
from model_service.scheduler.store.repository import ExecutionRepositoryUnavailableError


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import timedelta

    from model_service.providers.provider import ProviderAdapter
    from model_service.scheduler.builder_dispatch import HTTPBuilderDispatch
    from model_service.scheduler.internal_state import Settlement
    from model_service.scheduler.public import ModelResolver, ProviderName
    from model_service.scheduler.running_work import ActiveWorkRegistry
    from model_service.scheduler.store.repository import ExecutionRepository


class ExecutionWorker:
    """Claim due work, renew leases and settle."""

    def __init__(
        self,
        *,
        repository: ExecutionRepository,
        providers: Mapping[ProviderName, ProviderAdapter],
        dispatch: HTTPBuilderDispatch,
        resolve_model: ModelResolver,
        owner: str,
        lease_duration: timedelta,
        active_work: ActiveWorkRegistry,
        max_concurrent: int = 1,
    ) -> None:
        """Store collaborators and the number of attempts to run at once.

        Args:
            repository: Durable execution store.
            providers: Provider adapters by provider name.
            dispatch: Client that runs builders.
            resolve_model: Resolver of the current model binding for a builder and model.
            owner: Lease owner ID of this worker.
            lease_duration: How long a claim stays valid without renewal.
            active_work: Registry of local tasks, used for prompt cancellation.
            max_concurrent: Builder and provider attempts run at once.

        Raises:
            ValueError: max_concurrent is not positive.
        """
        if max_concurrent <= 0:
            raise ValueError("concurrency limits must be positive")
        self._repository = repository
        self._providers = dict(providers)
        self._dispatch = dispatch
        self._resolve_model = resolve_model
        self._owner = owner
        self._lease_duration = lease_duration
        self._active_work = active_work
        self._max_concurrent = max_concurrent

    async def run(self, stop: asyncio.Event, *, poll_interval: float) -> None:
        """Run maintenance and the claim loop until stop is set.

        Args:
            stop: Event that ends the loop; running attempts are then cancelled.
            poll_interval: Seconds between polls when no work was claimed.

        Raises:
            ValueError: poll_interval is not positive.
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
        """Run repository maintenance every poll interval until stop is set, logging failures.

        Args:
            stop: Event that ends the loop.
            poll_interval: Seconds between maintenance runs.
        """
        while not stop.is_set():
            try:
                await self._repository.maintain()
            except Exception as error:
                logger.error("execution maintenance failed: %s", type(error).__name__)
            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=poll_interval)

    async def _run_claims(self, stop: asyncio.Event, *, poll_interval: float) -> None:
        """Claim work into free slots and start a task per claim until stop is set.

        Claim failures are logged and the loop continues. On exit all running
        attempts are cancelled and awaited.

        Args:
            stop: Event that ends the loop.
            poll_interval: Seconds to wait when nothing was claimed and slots are free.
        """
        running: set[asyncio.Task[None]] = set()
        stopping = asyncio.ensure_future(stop.wait())
        try:
            while not stop.is_set():
                free = self._max_concurrent - len(running)
                claim = None
                if free > 0:
                    try:
                        claim = await self._repository.claim_due(
                            self._owner, self._lease_duration
                        )
                    except Exception as error:
                        logger.error("execution claim failed: %s", type(error).__name__)
                if claim is not None:
                    running.add(asyncio.create_task(self._process_logged(claim)))
                    continue
                await asyncio.wait(
                    {stopping, *running},
                    timeout=poll_interval if free > 0 else None,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                running.difference_update({task for task in running if task.done()})
        finally:
            stopping.cancel()
            pending = list(running)
            for task in pending:
                task.cancel()
            await asyncio.gather(stopping, *pending, return_exceptions=True)

    async def run_once(self) -> bool:
        """Claim and fully process one due attempt.

        Returns:
            True if an attempt was processed, False when nothing is due.

        Raises:
            ExecutionRepositoryUnavailableError: The repository is unavailable for the claim, or for the settlement during a whole lease.
        """
        claim = await self._repository.claim_due(self._owner, self._lease_duration)
        if claim is None:
            return False
        await self._process(claim)
        return True

    async def _process_logged(self, claim: WorkClaim) -> None:
        """Process a claim, logging any exception instead of raising it.

        Args:
            claim: Leased attempt.
        """
        try:
            await self._process(claim)
        except Exception as error:
            logger.error(
                "execution claim failed: id=%s kind=%s attempt=%s error=%s",
                claim.execution_id,
                claim.kind,
                claim.attempt,
                type(error).__name__,
            )

    async def _process(self, claim: WorkClaim) -> None:
        """Run one claim while renewing its lease; settle only a completed outcome.

        Nothing is settled when renewal stops the attempt or the execution was cancelled.

        Args:
            claim: Leased attempt.

        Raises:
            asyncio.CancelledError: The worker itself is being cancelled.
        """
        task = asyncio.create_task(self._execute(claim))
        self._active_work.register(claim.execution_id, claim.fencing_token, task)
        renewal_task = asyncio.create_task(self._renew_while_running(claim, task))
        try:
            done, _pending = await asyncio.wait(
                {task, renewal_task}, return_when=asyncio.FIRST_COMPLETED
            )
            if renewal_task in done:
                directive = await renewal_task
                if directive is not RenewDirective.CONTINUE:
                    return
            settlement = await task
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
            # The claim's own task was cancelled (execution cancellation): the
            # cancel transition already replaced its work, so there is nothing to settle.
            return
        finally:
            for pending_task in (task, renewal_task):
                if not pending_task.done():
                    pending_task.cancel()
            await asyncio.gather(task, renewal_task, return_exceptions=True)
            self._active_work.unregister(claim.execution_id, claim.fencing_token)
        await self._settle(claim, settlement)

    async def _settle(self, claim: WorkClaim, settlement: Settlement) -> None:
        """Record the outcome, retrying a failed write for up to one lease.

        Losing it would rerun finished work, or exhaust its attempts. A retry is
        safe: a settlement that did land, or a lease that was given up meanwhile,
        makes the claim stale, and a stale claim ends the retries quietly.

        Args:
            claim: Leased attempt.
            settlement: Outcome of the attempt.

        Raises:
            TimeoutError: Writes kept timing out for a whole lease.
            OSError: Writes kept failing for a whole lease.
            ExecutionRepositoryUnavailableError: The repository stayed unavailable for a whole lease.
        """
        loop = asyncio.get_running_loop()
        give_up_at = loop.time() + self._lease_duration.total_seconds()
        delay = 0.1
        while True:
            try:
                await self._repository.settle(
                    claim.execution_id, claim.fencing_token, settlement
                )
                return
            except StaleWorkClaimError:
                return
            except (
                TimeoutError,
                OSError,
                ExecutionRepositoryUnavailableError,
            ) as error:
                if loop.time() + delay >= give_up_at:
                    raise
                logger.warning(
                    "settlement failed, retrying: id=%s error=%s",
                    claim.execution_id,
                    type(error).__name__,
                )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 5.0)

    async def _renew_while_running(
        self, claim: WorkClaim, task: asyncio.Task[Settlement]
    ) -> RenewDirective:
        """Renew about every third of a lease while the attempt runs.

        A failed renewal is retried until the lease would have expired, so a short
        database outage does not abort long processing that still owns its lease.

        Args:
            claim: Leased attempt.
            task: Task running the attempt.

        Returns:
            CONTINUE once the task finishes, LEASE_LOST if renewals failed for a whole
            lease, or the first other directive the repository returns.
        """
        interval = max(self._lease_duration.total_seconds() / 3, 0.001)
        loop = asyncio.get_running_loop()
        renewed_at = loop.time()
        while not task.done():
            await asyncio.sleep(interval)
            if task.done():
                return RenewDirective.CONTINUE
            try:
                async with asyncio.timeout(interval):
                    directive = await self._repository.renew(
                        claim.execution_id, claim.fencing_token, self._lease_duration
                    )
            except (TimeoutError, ExecutionRepositoryUnavailableError) as error:
                if loop.time() - renewed_at >= self._lease_duration.total_seconds():
                    return RenewDirective.LEASE_LOST
                logger.warning(
                    "lease renewal failed, retrying: id=%s error=%s",
                    claim.execution_id,
                    type(error).__name__,
                )
                continue
            if directive is not RenewDirective.CONTINUE:
                return directive
            renewed_at = loop.time()
        return RenewDirective.CONTINUE

    async def _execute(self, claim: WorkClaim) -> Settlement:
        """Run the claimed work: builder processing, or a provider finish or report.

        Provider finish marks the mutation boundary before its first external write.

        Args:
            claim: Leased attempt.

        Returns:
            The attempt's settlement.

        Raises:
            RuntimeError: The claim lacks what its work kind needs.
        """
        if claim.kind is WorkKind.PROCESSING:
            return await self._run_builder(claim)
        try:
            mutation_started = False

            async def before_mutation() -> None:
                """Persist the mutation boundary once, before the first external write."""
                nonlocal mutation_started
                if mutation_started:
                    return
                await self._repository.mark_finish_mutation_started(
                    claim.execution_id, claim.fencing_token
                )
                mutation_started = True

            result = await self._providers[claim.provider].finish(
                claim.protected_context,
                _finish_outcome(claim),
                before_mutation=before_mutation,
            )
            return ProviderFinishSucceeded(result)
        except ProviderFinishFailureError as error:
            return ProviderFinishFailed(
                error=ExecutionError(error.code, error.safe_message),
                retryable=error.retryable,
                outcome_unknown=error.outcome_unknown,
            )

    async def _run_builder(self, claim: WorkClaim) -> Settlement:
        """Resolve the model's current binding and dispatch the builder.

        Args:
            claim: Leased attempt.

        Returns:
            Success with the builder result; a rejection for confirmed backpressure; or
            a failure, permanent for an unknown or incompatible model and retryable for
            an unknown dispatch outcome.
        """
        try:
            # Each attempt uses the model's current catalog declaration.
            model = self._resolve_model(claim.builder_id, claim.model_id)
        except (UnknownBuilderError, UnknownModelError):
            return ProcessingFailed(
                error=ExecutionError(
                    ExecutionErrorCode.UNKNOWN_MODEL,
                    f"{claim.builder_id}/{claim.model_id} is no longer deployed",
                ),
                retryable=False,
            )
        except IncompatibleModelError:
            return ProcessingFailed(
                error=ExecutionError(
                    ExecutionErrorCode.INCOMPATIBLE_MODEL,
                    f"model {claim.model_id!r} is no longer compatible with "
                    f"builder {claim.builder_id!r}",
                ),
                retryable=False,
            )
        try:
            result = await self._dispatch.run(
                claim.builder_id,
                claim.invocation,
                model,
                ExecutionContext(
                    processing_attempt_id=claim.fencing_token,
                    protected_context=claim.protected_context,
                ),
            )
            return ProcessingSucceeded(result)
        except ConfirmedRejectionError as error:
            return ProcessingRejected(error.retry_after)
        except UnknownOutcomeError:
            return ProcessingFailed(
                error=ExecutionError(
                    code=ExecutionErrorCode.BUILDER_OUTCOME_UNKNOWN,
                    message="builder dispatch outcome is unknown",
                ),
                retryable=True,
            )
        except RemoteFailureError as error:
            return ProcessingFailed(
                error=ExecutionError(error.code, error.safe_message),
                retryable=error.retryable,
            )


def _finish_outcome(claim: WorkClaim) -> ProviderFinishOutcome:
    """Convert a provider claim into success, failure, or cancellation.

    Args:
        claim: Leased attempt.

    Returns:
        The outcome for the provider to publish or report.

    Raises:
        RuntimeError: The claim lacks the stored result or error its kind requires, or its kind is not provider work.
    """
    if claim.kind is WorkKind.PROVIDER_FINISH:
        if not isinstance(claim.result, BuilderResult):
            raise RuntimeError("provider finish requires a durable builder result")
        return ProviderSuccess(claim.result, claim.builder_id)
    if claim.kind is WorkKind.PROVIDER_FAILURE_REPORT:
        if claim.error is None:
            raise RuntimeError("provider failure report requires a durable error")
        return ProviderFailure(claim.error)
    if claim.kind is WorkKind.PROVIDER_CANCELLATION_REPORT:
        return ProviderCancellation()
    raise RuntimeError("unsupported provider work kind")
