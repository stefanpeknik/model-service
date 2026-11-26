"""The execution repository: atomic admission, claims, settlement and maintenance.

Every rule lives here and in `transitions/` and `queue.py`; a storage only reads and
writes records. Provider contexts are encrypted here, so storage never holds them
in plain text, and dropped once an execution has no work left.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING
from uuid import uuid4

from model_service.control_plane.scheduler.internal_state import AdmissionResult
from model_service.control_plane.scheduler.public import ScheduleConflictError
from model_service.control_plane.scheduler.queue import WorkQueue
from model_service.control_plane.scheduler.transitions.execution_transitions import (
    ExecutionTransitions,
)


if TYPE_CHECKING:
    from collections.abc import Collection
    from datetime import timedelta

    from model_service.control_plane.scheduler.internal_state import (
        Admission,
        ExecutionRecord,
        RenewDirective,
        RetryPolicy,
        Settlement,
        WorkClaim,
    )
    from model_service.control_plane.scheduler.public import SubmissionIdentity
    from model_service.control_plane.scheduler.store.protected_context import (
        ProtectedContextCipher,
    )
    from model_service.control_plane.scheduler.store.storage import (
        ExecutionStorage,
        StorageUnit,
    )
    from model_service.control_plane.scheduler.targets import CapacityLimits


logger = logging.getLogger(__name__)


class ExecutionRepository:
    """Replay-safe admission, leased claims, settlement and maintenance over a storage."""

    def __init__(
        self,
        storage: ExecutionStorage,
        *,
        cipher: ProtectedContextCipher,
        capacity_limits: CapacityLimits,
        processing_policy: RetryPolicy,
        finish_policy: RetryPolicy,
    ) -> None:
        """Keep the storage and build the state machine over the retry policies.

        Args:
            storage: Where executions live.
            cipher: Cipher for provider contexts at rest.
            capacity_limits: Attempts allowed at once per capacity key (a model,
                or a model-free pipeline).
            processing_policy: Processing retry policy.
            finish_policy: Provider finish retry policy.
        """
        self._storage = storage
        self._cipher = cipher
        self._transitions = ExecutionTransitions(processing_policy, finish_policy)
        self._queue = WorkQueue(capacity_limits, self._transitions.expiry)

    async def initialize(self) -> None:
        """Prepare the storage before first use."""
        await self._storage.initialize()

    async def aclose(self) -> None:
        """Release the storage's resources."""
        await self._storage.aclose()

    async def check_ready(self) -> None:
        """Check that the storage answers.

        Raises:
            ExecutionRepositoryUnavailableError: Storage is unavailable.
        """
        await self._storage.check_ready()

    async def maintain(self) -> None:
        """Expire passed deadlines and recover abandoned leases, one execution at a time.

        Raises:
            ExecutionRepositoryUnavailableError: Storage is unavailable.
        """
        async with self._storage.unit() as unit:
            now = await unit.now()
            stale = [
                execution.id
                for execution in await unit.with_work()
                if self._queue.maintenance_change(execution, now) is not None
            ]
        for execution_id in stale:
            async with self._storage.unit() as unit:
                # Checked again under the lock: a worker may have settled it since.
                changed = self._queue.maintenance_change(
                    await unit.get(execution_id, lock=True), await unit.now()
                )
                if changed is not None:
                    await self._save(unit, changed)

    async def find_submission(
        self,
        identity: SubmissionIdentity,
    ) -> ExecutionRecord | None:
        """Find a provider-scoped replay key, returning None if absent and rejecting fingerprint conflicts.

        Args:
            identity: Replay identity of the submission.

        Returns:
            The existing execution, or None.

        Raises:
            ScheduleConflictError: The key was used for a request with a different fingerprint.
            ExecutionRepositoryUnavailableError: Storage is unavailable.
        """
        async with self._storage.unit() as unit:
            execution = await unit.find(identity)
        if execution is not None:
            _check_fingerprint(execution, identity)
        return execution

    async def admit(self, admission: Admission) -> AdmissionResult:
        """Atomically insert a prepared execution or return the existing identical submission.

        Args:
            admission: Prepared execution to store.

        Returns:
            The stored execution and whether this call created it.

        Raises:
            ScheduleConflictError: A concurrent submission used the key for a different request.
            ExecutionRepositoryUnavailableError: Storage is unavailable.
        """
        sealed = self._cipher.encrypt(admission.protected_context)
        async with self._storage.unit() as unit:
            execution = self._transitions.admission.admit(
                admission, str(uuid4()), await unit.now()
            )
            if await unit.insert(execution, sealed):
                return AdmissionResult(
                    replace(execution, protected_context={}), created=True
                )
            existing = await unit.find(admission.identity)
        if existing is None:
            raise RuntimeError("conflicting admission disappeared")
        _check_fingerprint(existing, admission.identity)
        return AdmissionResult(existing, created=False)

    async def get(self, execution_id: str) -> ExecutionRecord:
        """Return the record, without its protected context.

        Args:
            execution_id: ID of the execution.

        Returns:
            The record.

        Raises:
            UnknownExecutionError: No execution has this ID.
            ExecutionRepositoryUnavailableError: Storage is unavailable.
        """
        async with self._storage.unit() as unit:
            return await unit.get(execution_id)

    async def get_many(self, execution_ids: Collection[str]) -> list[ExecutionRecord]:
        """Return the records that exist, in the order requested.

        Args:
            execution_ids: IDs to read; unknown ones are skipped.

        Returns:
            The existing records in request order.

        Raises:
            ExecutionRepositoryUnavailableError: Storage is unavailable.
        """
        async with self._storage.unit() as unit:
            return await unit.get_many(execution_ids)

    async def request_cancel(self, execution_id: str) -> ExecutionRecord:
        """Record cancellation and invalidate the active claim, leaving terminal executions unchanged.

        Args:
            execution_id: ID of the execution to cancel.

        Returns:
            The record after the request.

        Raises:
            UnknownExecutionError: No execution has this ID.
            ExecutionRepositoryUnavailableError: Storage is unavailable.
        """
        async with self._storage.unit() as unit:
            execution = await unit.get(execution_id, lock=True)
            cancelled = self._transitions.admission.cancel(execution, await unit.now())
            if cancelled is not execution:
                await self._save(unit, cancelled)
            return cancelled

    async def claim_due(
        self,
        owner: str,
        lease_duration: timedelta,
    ) -> WorkClaim | None:
        """Lease the next attempt; see `WorkQueue.pick_claim` for the order.

        Claims take the storage's claim lock, so two never count the same free
        slot. An execution whose context cannot be decrypted fails in place and
        this call claims nothing.

        Args:
            owner: ID of the claiming worker.
            lease_duration: How long the claim stays valid without renewal.

        Returns:
            The claim, or None when nothing is due.

        Raises:
            ExecutionRepositoryUnavailableError: Storage is unavailable.
        """
        async with self._storage.unit() as unit:
            await unit.lock_claims()
            now = await unit.now()
            chosen = self._queue.pick_claim(await unit.with_work(), now)
            if chosen is None:
                return None
            execution = await unit.get(chosen.id, lock=True)
            if not self._queue.is_due(execution, now):
                return None
            try:
                context = self._cipher.decrypt(await unit.sealed_context(execution.id))
            except ValueError:  # Includes ProtectedContextError: a bad envelope or key.
                logger.error(
                    "execution %s has an unreadable provider context", execution.id
                )
                # The stored bytes stay, in case the key was wrong.
                await unit.save(self._transitions.outcomes.unreadable(execution, now))
                return None
            claimed = self._transitions.leases.claim(
                replace(execution, protected_context=context),
                owner=owner,
                fencing_token=str(uuid4()),
                lease_duration=lease_duration,
                now=now,
            )
            await self._save(unit, claimed.execution)
        return claimed.claim

    async def renew(
        self,
        execution_id: str,
        fencing_token: str,
        lease_duration: timedelta,
    ) -> RenewDirective:
        """Extend a current lease or return a stop directive when ownership or the work deadline is lost.

        Args:
            execution_id: ID of the claimed execution.
            fencing_token: Token of the renewing attempt.
            lease_duration: Extension to apply from now.

        Returns:
            The directive the worker must follow.

        Raises:
            UnknownExecutionError: No execution has this ID.
            ExecutionRepositoryUnavailableError: Storage is unavailable.
        """
        async with self._storage.unit() as unit:
            decision = self._transitions.leases.decide_renewal(
                await unit.get(execution_id, lock=True),
                fencing_token=fencing_token,
                lease_duration=lease_duration,
                now=await unit.now(),
            )
            if decision.execution is not None:
                await self._save(unit, decision.execution)
        return decision.directive

    async def settle(
        self,
        execution_id: str,
        fencing_token: str,
        settlement: Settlement,
    ) -> ExecutionRecord:
        """Apply an attempt's outcome only with a current token, unexpired lease, and valid deadline.

        A rejection still commits the state change a passed deadline brings.

        Args:
            execution_id: ID of the claimed execution.
            fencing_token: Token of the settling attempt.
            settlement: Outcome of the attempt.

        Returns:
            The record after the outcome.

        Raises:
            UnknownExecutionError: No execution has this ID.
            StaleWorkClaimError: The token or lease is no longer current, the deadline passed, or the work kind does not match.
            ExecutionRepositoryUnavailableError: Storage is unavailable.
        """
        async with self._storage.unit() as unit:
            decision = self._transitions.outcomes.decide_settlement(
                await unit.get(execution_id, lock=True),
                fencing_token,
                settlement,
                await unit.now(),
            )
            if decision.execution is not None:
                await self._save(unit, decision.execution)
        return decision.accepted_execution()

    async def mark_finish_mutation_started(
        self,
        execution_id: str,
        fencing_token: str,
    ) -> None:
        """Persist the mutation boundary before an external provider write so recovery knows its outcome may be uncertain.

        Args:
            execution_id: ID of the claimed execution.
            fencing_token: Token of the provider attempt.

        Raises:
            UnknownExecutionError: No execution has this ID.
            StaleWorkClaimError: The claim is not provider work or is no longer current.
            ExecutionRepositoryUnavailableError: Storage is unavailable.
        """
        async with self._storage.unit() as unit:
            marked = self._transitions.outcomes.mark_finish_mutation_started(
                await unit.get(execution_id, lock=True),
                fencing_token,
                await unit.now(),
            )
            await self._save(unit, marked)

    async def _save(self, unit: StorageUnit, execution: ExecutionRecord) -> None:
        """Write a record, dropping its context once no work will use the credential again."""
        await unit.save(
            execution,
            sealed_context=(
                self._cipher.encrypt({}) if execution.work_kind is None else None
            ),
        )


def _check_fingerprint(
    execution: ExecutionRecord, identity: SubmissionIdentity
) -> None:
    """Reject reuse of a provider/key identity with different request contents.

    Args:
        execution: Stored execution with that identity.
        identity: Identity of the new submission.

    Raises:
        ScheduleConflictError: The fingerprints differ.
    """
    if execution.identity.fingerprint != identity.fingerprint:
        raise ScheduleConflictError(
            "submission identity was reused with a different request"
        )
