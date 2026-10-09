"""Check repository claim ordering, workflow capacity, and fencing shared by both adapters."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from execution_scheduler.domain import (
    Admission,
    ProcessingRejected,
    ProcessingSucceeded,
    StaleWorkClaimError,
    SubmissionIdentity,
    WorkKind,
)
from workflow_execution.contracts import (
    InputAccessContext,
    WorkflowInvocation,
    WorkflowResult,
)


if TYPE_CHECKING:
    from execution_scheduler.repository import ExecutionRepository


class ExecutionRepositoryContract:
    """Exercise repository claim ordering, workflow capacity, and fencing shared by both adapters."""

    repository: ExecutionRepository

    async def _advance_time(self, duration: timedelta) -> None:
        """Advance the repository test clock or wait for database time according to the concrete adapter.

        Args:
            duration: Amount of time to advance or wait in a deterministic lifecycle
                scenario.

        Returns:
            None: No return value.

        Raises:
            NotImplementedError: The concrete repository contract fixture has not
                supplied its clock hook.
        """
        raise NotImplementedError

    async def _admit_execution(self, key: str, workflow_id: str = "heatmap") -> str:
        """Admit deferred standalone work with a destination and return its durable ID.

        Args:
            key: Caller replay key identifying this fixture submission.
            workflow_id: Declared workflow identifier used for catalog lookup and
                dispatch routing.

        Returns:
            str: Durable ID of the minimal admitted repository fixture.
        """
        deadline = datetime.now(UTC) + timedelta(minutes=5)
        admitted = await self.repository.admit(
            Admission(
                identity=SubmissionIdentity("standalone", key, key),
                workflow_id=workflow_id,
                invocation=WorkflowInvocation(
                    inputs={},
                    resolved_configuration=None,
                    input_access=InputAccessContext(
                        kind="shared_storage", state={"root": "/shared"}
                    ),
                    deadline_at=deadline,
                    output_path="outputs/requested.tiff",
                ),
                protected_context={},
                deadline_at=deadline,
            )
        )
        return admitted.execution.id

    async def test_due_provider_finish_precedes_older_processing(self) -> None:
        """Verify due provider finish precedes older processing.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        older_id = await self._admit_execution("older", "older-workflow")
        older_claim = await self.repository.claim_due("worker", timedelta(seconds=30))
        assert older_claim is not None
        await self._advance_time(timedelta(milliseconds=10))
        newer_id = await self._admit_execution("newer", "newer-workflow")
        newer_claim = await self.repository.claim_due("worker", timedelta(seconds=30))
        assert newer_claim is not None
        assert newer_claim.execution_id == newer_id
        await self.repository.settle(
            newer_id, newer_claim.fencing_token, ProcessingSucceeded(WorkflowResult({}))
        )
        await self.repository.settle(
            older_id, older_claim.fencing_token, ProcessingRejected(timedelta(0))
        )

        selected = await self.repository.claim_due("worker", timedelta(seconds=30))

        assert selected is not None
        assert selected.execution_id == newer_id, selected
        assert selected.kind is WorkKind.PROVIDER_FINISH

    async def test_processing_within_workflow_orders_by_due_time(self) -> None:
        """Verify processing within workflow orders by due time.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        older_id = await self._admit_execution("older")
        older_claim = await self.repository.claim_due("worker", timedelta(seconds=30))
        assert older_claim is not None
        await self.repository.settle(
            older_id,
            older_claim.fencing_token,
            ProcessingRejected(timedelta(milliseconds=200)),
        )
        await self._advance_time(timedelta(milliseconds=10))
        newer_id = await self._admit_execution("newer")
        await self._advance_time(timedelta(milliseconds=300))

        selected = await self.repository.claim_due("worker", timedelta(seconds=30))

        assert selected is not None
        assert selected.execution_id == newer_id, selected

    async def test_provider_finish_orders_by_due_time_before_creation(self) -> None:
        """Verify provider finish orders by due time before creation.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        older_id = await self._admit_execution("older", "older-workflow")
        older = await self.repository.claim_due("worker", timedelta(seconds=30))
        assert older is not None
        await self._advance_time(timedelta(milliseconds=10))
        newer_id = await self._admit_execution("newer", "newer-workflow")
        newer = await self.repository.claim_due("worker", timedelta(seconds=30))
        assert newer is not None
        await self.repository.settle(
            newer_id, newer.fencing_token, ProcessingSucceeded(WorkflowResult({}))
        )
        await self._advance_time(timedelta(milliseconds=10))
        await self.repository.settle(
            older_id, older.fencing_token, ProcessingSucceeded(WorkflowResult({}))
        )

        selected = await self.repository.claim_due("worker", timedelta(seconds=30))

        assert selected is not None
        assert selected.execution_id == newer_id, selected
        assert selected.kind is WorkKind.PROVIDER_FINISH

    async def test_retry_not_yet_due_does_not_block_new_processing(self) -> None:
        """Verify retry not yet due does not block new processing.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        older_id = await self._admit_execution("older")
        claim = await self.repository.claim_due("worker", timedelta(seconds=30))
        assert claim is not None
        await self.repository.settle(
            older_id, claim.fencing_token, ProcessingRejected(timedelta(minutes=1))
        )
        await self._advance_time(timedelta(milliseconds=10))
        newer_id = await self._admit_execution("newer")

        selected = await self.repository.claim_due("worker", timedelta(seconds=30))

        assert selected is not None
        assert selected.execution_id == newer_id, selected

    async def test_processing_workflows_order_by_oldest_eligible_creation(self) -> None:
        """Verify processing workflows order by oldest eligible creation.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        older_id = await self._admit_execution("older", "older-workflow")
        older_claim = await self.repository.claim_due("worker", timedelta(seconds=30))
        assert older_claim is not None
        await self.repository.settle(
            older_id,
            older_claim.fencing_token,
            ProcessingRejected(timedelta(milliseconds=200)),
        )
        await self._advance_time(timedelta(milliseconds=10))
        await self._admit_execution("newer", "newer-workflow")
        await self._advance_time(timedelta(milliseconds=300))

        selected = await self.repository.claim_due("worker", timedelta(seconds=30))

        assert selected is not None
        assert selected.execution_id == older_id, selected

    async def test_workflow_at_capacity_does_not_block_another_workflow(self) -> None:
        """Verify workflow at capacity does not block another workflow.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        await self._admit_execution("active")
        active = await self.repository.claim_due("worker", timedelta(seconds=30))
        assert active is not None
        await self._advance_time(timedelta(milliseconds=10))
        await self._admit_execution("blocked")
        await self._advance_time(timedelta(milliseconds=10))
        available_id = await self._admit_execution("available", "another-workflow")

        selected = await self.repository.claim_due("worker", timedelta(seconds=30))

        assert selected is not None
        assert selected.execution_id == available_id, selected

    async def test_terminal_report_precedes_older_processing(self) -> None:
        """Verify terminal report precedes older processing.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        await self._admit_execution("older")
        await self._advance_time(timedelta(milliseconds=10))
        cancelled_id = await self._admit_execution("cancelled")
        await self.repository.request_cancel(cancelled_id)

        selected = await self.repository.claim_due("worker", timedelta(seconds=30))

        assert selected is not None
        assert selected.execution_id == cancelled_id, selected
        assert selected.kind is WorkKind.PROVIDER_CANCELLATION_REPORT

    async def test_cancellation_fences_late_processing_result(self) -> None:
        """Verify cancellation fences late processing result.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        execution_id = await self._admit_execution("cancelled")
        claim = await self.repository.claim_due("worker", timedelta(seconds=30))
        assert claim is not None
        await self.repository.request_cancel(execution_id)
        try:
            await self.repository.settle(
                execution_id,
                claim.fencing_token,
                ProcessingSucceeded(WorkflowResult({})),
            )
        except StaleWorkClaimError:
            pass
        else:
            raise AssertionError("cancelled processing claim accepted a late result")
        execution = await self.repository.get(execution_id)
        assert execution.result is None
        assert execution.work_kind is WorkKind.PROVIDER_CANCELLATION_REPORT
