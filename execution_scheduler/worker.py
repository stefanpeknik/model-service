"""Run leased processing and provider attempts with independent maintenance and renewal.

Results are settled before provider finish is claimed, so a failed publication is
retried without rerunning the builder.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress
from typing import TYPE_CHECKING

from builders.client import (
    ConfirmedRejectionError,
    RemoteFailureError,
    UnknownOutcomeError,
)
from builders.contracts import BuilderResult, ExecutionContext
from execution_scheduler import inference
from execution_scheduler.contracts import (
    ExecutionError,
    ExecutionErrorCode,
    IncompatibleModelError,
    InferenceInvocation,
    Operation,
    UnknownBuilderError,
    UnknownModelError,
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
from execution_scheduler.repository import ExecutionRepositoryUnavailableError


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import timedelta

    from builders.client import BuilderDispatch
    from execution_scheduler.active_work import ActiveWorkRegistry
    from execution_scheduler.contracts import ModelResolver
    from execution_scheduler.domain import Settlement
    from execution_scheduler.inference import InferenceTarget
    from execution_scheduler.providers import ProviderAdapter
    from execution_scheduler.repository import ExecutionRepository


class ExecutionWorker:
    """Claim due work, renew leases and settle.

    Builder work (with its provider phases) and raw inference have separate slot
    pools, `max_concurrent` and `max_concurrent_inference`, so a burst of tiles
    cannot starve builders or the other way round.
    """

    def __init__(
        self,
        *,
        repository: ExecutionRepository,
        providers: Mapping[str, ProviderAdapter],
        dispatch: BuilderDispatch,
        resolve_model: ModelResolver,
        owner: str,
        lease_duration: timedelta,
        active_work: ActiveWorkRegistry,
        max_concurrent: int = 1,
        inference_target: InferenceTarget | None = None,
        max_concurrent_inference: int = 0,
    ) -> None:
        if max_concurrent <= 0 or max_concurrent_inference < 0:
            raise ValueError("concurrency limits must be positive")
        self._repository = repository
        self._providers = dict(providers)
        self._dispatch = dispatch
        self._resolve_model = resolve_model
        self._owner = owner
        self._lease_duration = lease_duration
        self._active_work = active_work
        self._inference = inference_target
        self._slots = {
            Operation.BUILDER: max_concurrent,
            Operation.INFERENCE: (
                max_concurrent_inference if inference_target is not None else 0
            ),
        }

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
            except Exception as error:
                logger.error("execution maintenance failed: %s", type(error).__name__)
            with suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=poll_interval)

    async def _run_claims(self, stop: asyncio.Event, *, poll_interval: float) -> None:
        running: dict[Operation, set[asyncio.Task[None]]] = {
            operation: set() for operation in Operation
        }
        stopping = asyncio.ensure_future(stop.wait())
        try:
            while not stop.is_set():
                free = [
                    operation
                    for operation, tasks in running.items()
                    if len(tasks) < self._slots[operation]
                ]
                claim = None
                if free:
                    try:
                        claim = await self._repository.claim_due(
                            self._owner, self._lease_duration, free
                        )
                    except Exception as error:
                        logger.error("execution claim failed: %s", type(error).__name__)
                if claim is not None:
                    running[claim.operation].add(
                        asyncio.create_task(self._process_logged(claim))
                    )
                    continue
                await asyncio.wait(
                    {stopping, *(task for tasks in running.values() for task in tasks)},
                    timeout=poll_interval if free else None,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for tasks in running.values():
                    tasks.difference_update({task for task in tasks if task.done()})
        finally:
            stopping.cancel()
            pending = [task for tasks in running.values() for task in tasks]
            for task in pending:
                task.cancel()
            await asyncio.gather(stopping, *pending, return_exceptions=True)

    async def run_once(self) -> bool:
        """Claim and fully process one due attempt; False when nothing is due."""
        claim = await self._repository.claim_due(
            self._owner,
            self._lease_duration,
            [operation for operation, slots in self._slots.items() if slots],
        )
        if claim is None:
            return False
        await self._process(claim)
        return True

    async def _process_logged(self, claim: WorkClaim) -> None:
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
        """Run one claim while renewing its lease; settle only a completed outcome."""
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
        with suppress(StaleWorkClaimError):
            await self._repository.settle(
                claim.execution_id, claim.fencing_token, settlement
            )

    async def _renew_while_running(
        self, claim: WorkClaim, task: asyncio.Task[Settlement]
    ) -> RenewDirective:
        """Renew about every third of a lease.

        A failed renewal is retried until the lease would have expired, so a short
        database outage does not abort long processing that still owns its lease.
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
        if claim.kind is WorkKind.PROCESSING:
            if claim.operation is Operation.INFERENCE:
                return await self._run_inference(claim)
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

    async def _run_inference(self, claim: WorkClaim) -> Settlement:
        if (
            self._inference is None
            or claim.payload is None
            or not isinstance(claim.invocation, InferenceInvocation)
        ):
            raise RuntimeError("inference claim cannot run in this worker")
        try:
            # The model may have changed since admission.
            self._inference.check(claim.model_id, claim.payload)
        except UnknownModelError:
            return ProcessingFailed(
                error=ExecutionError(
                    ExecutionErrorCode.UNKNOWN_MODEL,
                    f"model {claim.model_id!r} is no longer deployed",
                ),
                retryable=False,
            )
        except inference.IncompatibleInputError as error:
            return ProcessingFailed(
                error=ExecutionError(ExecutionErrorCode.INCOMPATIBLE_INPUT, str(error)),
                retryable=False,
            )
        try:
            response = await self._inference.call(
                claim.model_id, claim.payload, claim.invocation.headers
            )
        except inference.ModelUnavailableError:
            return ProcessingFailed(
                error=ExecutionError(
                    ExecutionErrorCode.MODEL_UNAVAILABLE,
                    "model did not answer the request",
                ),
                retryable=True,
            )
        except inference.ResultTooLargeError:
            return ProcessingFailed(
                error=ExecutionError(
                    ExecutionErrorCode.RESULT_TOO_LARGE,
                    "model response exceeds the stored result limit",
                ),
                retryable=False,
            )
        return inference.settlement(response)

    async def _run_builder(self, claim: WorkClaim) -> Settlement:
        if claim.builder_id is None or isinstance(
            claim.invocation, InferenceInvocation
        ):
            raise RuntimeError("builder claim has no builder invocation")
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
    """Convert a provider claim into success, failure, or cancellation, requiring the corresponding stored result or error."""
    if claim.kind is WorkKind.PROVIDER_FINISH:
        if not isinstance(claim.result, BuilderResult):
            raise RuntimeError("provider finish requires a durable builder result")
        return ProviderSuccess(claim.result)
    if claim.kind is WorkKind.PROVIDER_FAILURE_REPORT:
        if claim.error is None:
            raise RuntimeError("provider failure report requires a durable error")
        return ProviderFailure(claim.error)
    if claim.kind is WorkKind.PROVIDER_CANCELLATION_REPORT:
        return ProviderCancellation()
    raise RuntimeError("unsupported provider work kind")
