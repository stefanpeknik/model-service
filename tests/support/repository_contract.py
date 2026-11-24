"""Claim ordering, per-model capacity and fencing shared by both repositories."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from model_service.contracts.invocation import InputAccessContext, Invocation
from model_service.contracts.result import RunResult, ServedModelRun, SharedArtifact
from model_service.control_plane.scheduler.internal_state import (
    Admission,
    ProcessingRejected,
    ProcessingSucceeded,
    ProviderFinishSucceeded,
    StaleWorkClaimError,
    WorkClaim,
    WorkKind,
)
from model_service.control_plane.scheduler.public import (
    Priority,
    SubmissionIdentity,
)
from tests.support.scheduling import stored_target


if TYPE_CHECKING:
    from model_service.control_plane.scheduler.store.repository import (
        ExecutionRepository,
    )


class ExecutionRepositoryContract:
    repository: ExecutionRepository

    async def _advance_time(self, duration: timedelta) -> None:
        raise NotImplementedError

    async def _admit_execution(
        self,
        key: str,
        model_id: str = "model-a",
        analysis_id: str = "analysis-a",
        priority: int = Priority.NORMAL,
        *,
        models: tuple[str, ...] | None = None,
    ) -> str:
        deadline = datetime.now(UTC) + timedelta(minutes=5)
        admitted = await self.repository.admit(
            Admission(
                identity=SubmissionIdentity("standalone", key, key),
                target=stored_target(
                    analysis_id, (model_id,) if models is None else models
                ),
                invocation=Invocation(
                    inputs={},
                    configuration=None,
                    input_access=InputAccessContext(
                        kind="shared_storage", state={"root": "/shared"}
                    ),
                    deadline_at=deadline,
                    output_path="outputs/requested.tiff",
                ),
                protected_context={},
                deadline_at=deadline,
                priority=priority,
            )
        )
        return admitted.execution.id

    async def test_due_provider_finish_precedes_older_processing(self) -> None:
        older_id = await self._admit_execution("older", "older-model")
        older_claim = await self.repository.claim_due("worker", timedelta(seconds=30))
        assert older_claim is not None
        await self._advance_time(timedelta(milliseconds=10))
        newer_id = await self._admit_execution("newer", "newer-model")
        newer_claim = await self.repository.claim_due("worker", timedelta(seconds=30))
        assert newer_claim is not None
        assert newer_claim.execution_id == newer_id
        await self.repository.settle(
            newer_id, newer_claim.fencing_token, ProcessingSucceeded(RunResult({}))
        )
        await self.repository.settle(
            older_id, older_claim.fencing_token, ProcessingRejected(timedelta(0))
        )

        selected = await self.repository.claim_due("worker", timedelta(seconds=30))

        assert selected is not None
        assert selected.execution_id == newer_id, selected
        assert selected.kind is WorkKind.PROVIDER_FINISH

    async def test_higher_priority_is_claimed_first(self) -> None:
        await self._admit_execution("older")
        await self._advance_time(timedelta(milliseconds=10))
        urgent_id = await self._admit_execution("urgent", priority=Priority.HIGH)

        selected = await self.repository.claim_due("worker", timedelta(seconds=30))

        assert selected is not None and selected.execution_id == urgent_id, selected
        assert (await self.repository.get(urgent_id)).priority == Priority.HIGH

    async def test_a_model_with_urgent_work_is_served_before_older_models(
        self,
    ) -> None:
        await self._admit_execution("older", "older-model")
        await self._advance_time(timedelta(milliseconds=10))
        await self._admit_execution("normal", "busy-model")
        await self._advance_time(timedelta(milliseconds=10))
        urgent_id = await self._admit_execution(
            "urgent", "busy-model", priority=Priority.HIGH
        )

        selected = await self.repository.claim_due("worker", timedelta(seconds=30))

        assert selected is not None and selected.execution_id == urgent_id, selected

    async def test_processing_within_model_orders_by_due_time(self) -> None:
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
        older_id = await self._admit_execution("older", "older-model")
        older = await self.repository.claim_due("worker", timedelta(seconds=30))
        assert older is not None
        await self._advance_time(timedelta(milliseconds=10))
        newer_id = await self._admit_execution("newer", "newer-model")
        newer = await self.repository.claim_due("worker", timedelta(seconds=30))
        assert newer is not None
        await self.repository.settle(
            newer_id, newer.fencing_token, ProcessingSucceeded(RunResult({}))
        )
        await self._advance_time(timedelta(milliseconds=10))
        await self.repository.settle(
            older_id, older.fencing_token, ProcessingSucceeded(RunResult({}))
        )

        selected = await self.repository.claim_due("worker", timedelta(seconds=30))

        assert selected is not None
        assert selected.execution_id == newer_id, selected
        assert selected.kind is WorkKind.PROVIDER_FINISH

    async def test_retry_not_yet_due_does_not_block_new_processing(self) -> None:
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

    async def test_processing_models_order_by_oldest_eligible_creation(self) -> None:
        older_id = await self._admit_execution("older", "older-model")
        older_claim = await self.repository.claim_due("worker", timedelta(seconds=30))
        assert older_claim is not None
        await self.repository.settle(
            older_id,
            older_claim.fencing_token,
            ProcessingRejected(timedelta(milliseconds=200)),
        )
        await self._advance_time(timedelta(milliseconds=10))
        await self._admit_execution("newer", "newer-model")
        await self._advance_time(timedelta(milliseconds=300))

        selected = await self.repository.claim_due("worker", timedelta(seconds=30))

        assert selected is not None
        assert selected.execution_id == older_id, selected

    async def test_model_at_capacity_does_not_block_another_model(self) -> None:
        await self._admit_execution("active")
        active = await self.repository.claim_due("worker", timedelta(seconds=30))
        assert active is not None
        await self._advance_time(timedelta(milliseconds=10))
        await self._admit_execution("blocked")
        await self._advance_time(timedelta(milliseconds=10))
        available_id = await self._admit_execution("available", "another-model")

        selected = await self.repository.claim_due("worker", timedelta(seconds=30))

        assert selected is not None
        assert selected.execution_id == available_id, selected

    async def test_analyses_sharing_a_model_share_its_capacity(self) -> None:
        await self._admit_execution("active", analysis_id="first")
        active = await self.repository.claim_due("worker", timedelta(seconds=30))
        assert active is not None
        await self._advance_time(timedelta(milliseconds=10))
        await self._admit_execution("other-analysis", analysis_id="second")

        selected = await self.repository.claim_due("worker", timedelta(seconds=30))

        assert selected is None, selected

    async def test_a_multi_model_run_waits_for_a_slot_in_every_model(self) -> None:
        await self._admit_execution("active", "model-b")
        assert await self.repository.claim_due("worker", timedelta(seconds=30))
        await self._advance_time(timedelta(milliseconds=10))
        await self._admit_execution("both", models=("model-a", "model-b"))
        await self._advance_time(timedelta(milliseconds=10))
        single_id = await self._admit_execution("single", "model-a")

        selected = await self.repository.claim_due("worker", timedelta(seconds=30))

        assert selected is not None and selected.execution_id == single_id, selected

    async def test_a_multi_model_run_takes_a_slot_in_each_model(self) -> None:
        both_id = await self._admit_execution("both", models=("model-b", "model-a"))
        claimed = await self.repository.claim_due("worker", timedelta(seconds=30))
        assert claimed is not None and claimed.execution_id == both_id, claimed
        await self._admit_execution("first", "model-a")
        await self._admit_execution("second", "model-b")

        selected = await self.repository.claim_due("worker", timedelta(seconds=30))

        assert selected is None, selected

    async def test_model_free_runs_are_counted_by_their_analysis(self) -> None:
        await self._admit_execution("active", analysis_id="free", models=())
        assert await self.repository.claim_due("worker", timedelta(seconds=30))
        await self._advance_time(timedelta(milliseconds=10))
        await self._admit_execution("same", analysis_id="free", models=())
        await self._advance_time(timedelta(milliseconds=10))
        other_id = await self._admit_execution("other", analysis_id="also", models=())
        await self._advance_time(timedelta(milliseconds=10))
        model_id = await self._admit_execution("model", analysis_id="free")

        first = await self.repository.claim_due("worker", timedelta(seconds=30))
        second = await self.repository.claim_due("worker", timedelta(seconds=30))
        third = await self.repository.claim_due("worker", timedelta(seconds=30))

        assert first is not None and first.execution_id == other_id, first
        assert second is not None and second.execution_id == model_id, second
        assert third is None, third

    async def test_terminal_report_precedes_older_processing(self) -> None:
        await self._admit_execution("older")
        await self._advance_time(timedelta(milliseconds=10))
        cancelled_id = await self._admit_execution("cancelled")
        await self.repository.request_cancel(cancelled_id)

        selected = await self.repository.claim_due("worker", timedelta(seconds=30))

        assert selected is not None
        assert selected.execution_id == cancelled_id, selected
        assert selected.kind is WorkKind.PROVIDER_CANCELLATION_REPORT

    async def test_cancellation_fences_late_processing_result(self) -> None:
        execution_id = await self._admit_execution("cancelled")
        claim = await self.repository.claim_due("worker", timedelta(seconds=30))
        assert claim is not None
        await self.repository.request_cancel(execution_id)
        try:
            await self.repository.settle(
                execution_id,
                claim.fencing_token,
                ProcessingSucceeded(RunResult({})),
            )
        except StaleWorkClaimError:
            pass
        else:
            raise AssertionError("cancelled processing claim accepted a late result")
        execution = await self.repository.get(execution_id)
        assert execution.result is None
        assert execution.work_kind is WorkKind.PROVIDER_CANCELLATION_REPORT

    async def test_a_cancellation_report_survives_a_cancelled_publication(
        self,
    ) -> None:
        execution_id = await self._admit_execution("cancelled-in-finish")
        processing = await self.repository.claim_due("worker", timedelta(seconds=30))
        assert processing is not None
        await self.repository.settle(
            execution_id,
            processing.fencing_token,
            ProcessingSucceeded(RunResult({})),
        )
        finish = await self.repository.claim_due("worker", timedelta(seconds=30))
        assert finish is not None
        assert finish.kind is WorkKind.PROVIDER_FINISH
        await self.repository.mark_finish_mutation_started(
            execution_id, finish.fencing_token
        )
        await self.repository.request_cancel(execution_id)
        report = await self.repository.claim_due("worker", timedelta(milliseconds=30))
        assert report is not None
        assert report.kind is WorkKind.PROVIDER_CANCELLATION_REPORT

        # The report's worker dies before it writes anything.
        await self._advance_time(timedelta(milliseconds=80))
        await self.repository.maintain()
        retried = await self.repository.claim_due("worker", timedelta(seconds=30))

        assert retried is not None, "the cancellation report was dropped"
        assert retried.execution_id == execution_id
        assert retried.kind is WorkKind.PROVIDER_CANCELLATION_REPORT

    async def _lose_the_lease(self, kind: WorkKind) -> WorkClaim:
        """Claim due work with a short lease, then let the worker die."""
        claim = await self.repository.claim_due("worker", timedelta(milliseconds=30))
        assert claim is not None
        assert claim.kind is kind, claim
        await self._advance_time(timedelta(milliseconds=80))
        await self.repository.maintain()
        return claim

    async def _succeed_processing(self, execution_id: str) -> None:
        processing = await self.repository.claim_due("worker", timedelta(seconds=30))
        assert processing is not None
        await self.repository.settle(
            execution_id, processing.fencing_token, ProcessingSucceeded(RunResult({}))
        )

    async def _failure_code(self, execution_id: str) -> tuple[str, str | None]:
        execution = await self.repository.get(execution_id)
        return execution.status, execution.error.code if execution.error else None

    async def test_processing_that_keeps_losing_its_lease_exhausts_its_attempts(
        self,
    ) -> None:
        execution_id = await self._admit_execution("lost-processing")

        for _ in range(3):
            await self._lose_the_lease(WorkKind.PROCESSING)

        assert await self._failure_code(execution_id) == (
            "failed",
            "processing_attempts_exhausted",
        )

    async def test_a_finish_that_keeps_losing_its_lease_exhausts_its_attempts(
        self,
    ) -> None:
        execution_id = await self._admit_execution("lost-finish")
        await self._succeed_processing(execution_id)

        for _ in range(3):
            await self._lose_the_lease(WorkKind.PROVIDER_FINISH)

        assert await self._failure_code(execution_id) == (
            "failed",
            "provider_finish_attempts_exhausted",
        )

    async def test_a_publication_interrupted_midway_has_an_unknown_outcome(
        self,
    ) -> None:
        execution_id = await self._admit_execution("interrupted-finish")
        await self._succeed_processing(execution_id)
        finish = await self.repository.claim_due("worker", timedelta(milliseconds=30))
        assert finish is not None
        await self.repository.mark_finish_mutation_started(
            execution_id, finish.fencing_token
        )

        await self._advance_time(timedelta(milliseconds=80))
        await self.repository.maintain()

        assert await self._failure_code(execution_id) == (
            "failed",
            "provider_finish_outcome_unknown",
        )

    async def test_what_the_served_models_ran_outlives_a_moved_result(self) -> None:
        execution_id = await self._admit_execution("provenance")
        ran = ServedModelRun(
            digest="d" * 64,
            weights_revision="paige-ai/Virchow2@abc123",
            execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        processing = await self.repository.claim_due("worker", timedelta(seconds=30))
        assert processing is not None
        await self.repository.settle(
            execution_id,
            processing.fencing_token,
            ProcessingSucceeded(
                RunResult(
                    {"heatmap": SharedArtifact("/shared/.attempts/1/heatmap.npz")},
                    served_models={"model-a": ran},
                )
            ),
        )
        finish = await self.repository.claim_due("worker", timedelta(seconds=30))
        assert finish is not None

        # The provider moved the output and answers with its new path only.
        moved = RunResult({"heatmap": SharedArtifact("/shared/heatmap.npz")})
        await self.repository.settle(
            execution_id, finish.fencing_token, ProviderFinishSucceeded(moved)
        )

        execution = await self.repository.get(execution_id)
        assert execution.status == "succeeded"
        assert execution.result == RunResult(
            {"heatmap": SharedArtifact("/shared/heatmap.npz")},
            served_models={"model-a": ran},
        )

    async def test_get_many_keeps_order_and_skips_unknown_ids(self) -> None:
        first = await self._admit_execution("first", "model-b")
        second = await self._admit_execution("second")

        found = await self.repository.get_many(
            [second, "not-an-id", "00000000-0000-0000-0000-000000000000", first]
        )

        assert [execution.id for execution in found] == [second, first]
