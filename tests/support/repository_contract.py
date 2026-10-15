"""Claim ordering, per-model capacity and fencing shared by both repositories."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from model_service.builders.contracts import (
    BuilderResult,
    InputAccessContext,
    Invocation,
)
from model_service.scheduler.contracts import (
    ExecutionError,
    ExecutionStatus,
    InferenceInvocation,
    InferenceResult,
    Operation,
    ResultUnavailableError,
    UnknownExecutionError,
)
from model_service.scheduler.domain import (
    Admission,
    InferenceSucceeded,
    ProcessingFailed,
    ProcessingRejected,
    ProcessingSucceeded,
    StaleWorkClaimError,
    SubmissionIdentity,
    WorkKind,
)


if TYPE_CHECKING:
    from model_service.scheduler.store.repository import ExecutionRepository


INFERENCE_RETENTION = timedelta(milliseconds=300)
INFERENCE_LIMIT = 2


class ExecutionRepositoryContract:
    """Fixtures use INFERENCE_RETENTION and a per-model inference limit of INFERENCE_LIMIT."""

    repository: ExecutionRepository

    async def _advance_time(self, duration: timedelta) -> None:
        raise NotImplementedError

    async def _admit_execution(
        self, key: str, model_id: str = "model-a", builder_id: str = "heatmap"
    ) -> str:
        deadline = datetime.now(UTC) + timedelta(minutes=5)
        admitted = await self.repository.admit(
            Admission(
                identity=SubmissionIdentity("standalone", key, key),
                builder_id=builder_id,
                model_id=model_id,
                invocation=Invocation(
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

    async def _admit_inference(
        self, key: str, model_id: str = "model-a", payload: bytes = b"tile"
    ) -> str:
        deadline = datetime.now(UTC) + timedelta(minutes=5)
        admitted = await self.repository.admit(
            Admission(
                identity=SubmissionIdentity("inference", key, key),
                builder_id=None,
                model_id=model_id,
                invocation=InferenceInvocation(
                    headers={"x-output-dtype": "float32"}, deadline_at=deadline
                ),
                protected_context={},
                deadline_at=deadline,
                operation=Operation.INFERENCE,
                payload=payload,
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
            newer_id, newer_claim.fencing_token, ProcessingSucceeded(BuilderResult({}))
        )
        await self.repository.settle(
            older_id, older_claim.fencing_token, ProcessingRejected(timedelta(0))
        )

        selected = await self.repository.claim_due("worker", timedelta(seconds=30))

        assert selected is not None
        assert selected.execution_id == newer_id, selected
        assert selected.kind is WorkKind.PROVIDER_FINISH

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
            newer_id, newer.fencing_token, ProcessingSucceeded(BuilderResult({}))
        )
        await self._advance_time(timedelta(milliseconds=10))
        await self.repository.settle(
            older_id, older.fencing_token, ProcessingSucceeded(BuilderResult({}))
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

    async def test_builders_sharing_a_model_share_its_capacity(self) -> None:
        await self._admit_execution("active", builder_id="heatmap")
        active = await self.repository.claim_due("worker", timedelta(seconds=30))
        assert active is not None
        await self._advance_time(timedelta(milliseconds=10))
        await self._admit_execution("other-builder", builder_id="tile-embeddings")

        selected = await self.repository.claim_due("worker", timedelta(seconds=30))

        assert selected is None, selected

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
                ProcessingSucceeded(BuilderResult({})),
            )
        except StaleWorkClaimError:
            pass
        else:
            raise AssertionError("cancelled processing claim accepted a late result")
        execution = await self.repository.get(execution_id)
        assert execution.result is None
        assert execution.work_kind is WorkKind.PROVIDER_CANCELLATION_REPORT

    async def test_inference_claim_carries_payload_and_settles_without_provider(
        self,
    ) -> None:
        execution_id = await self._admit_inference("tile-1", payload=b"tile-bytes")
        claim = await self.repository.claim_due("worker", timedelta(seconds=30))
        assert claim is not None
        assert claim.operation is Operation.INFERENCE
        assert claim.payload == b"tile-bytes"
        assert claim.builder_id is None
        result = InferenceResult(
            "application/octet-stream", {"x-output-shape": "(2,)"}, 3
        )

        await self.repository.settle(
            execution_id, claim.fencing_token, InferenceSucceeded(result, b"out")
        )

        execution = await self.repository.get(execution_id)
        assert execution.status is ExecutionStatus.SUCCEEDED
        assert execution.work_kind is None
        assert execution.result == result
        assert await self.repository.inference_content(execution_id) == b"out"
        assert await self.repository.claim_due("worker", timedelta(seconds=30)) is None

    async def test_inference_rejects_a_builder_result(self) -> None:
        execution_id = await self._admit_inference("tile-1")
        claim = await self.repository.claim_due("worker", timedelta(seconds=30))
        assert claim is not None
        try:
            await self.repository.settle(
                execution_id,
                claim.fencing_token,
                ProcessingSucceeded(BuilderResult({})),
            )
        except StaleWorkClaimError:
            pass
        else:
            raise AssertionError("inference accepted a builder result")

    async def test_inference_capacity_is_per_model_and_separate_from_builders(
        self,
    ) -> None:
        await self._admit_execution("builder")
        for index in range(INFERENCE_LIMIT + 1):
            await self._admit_inference(f"tile-{index}")
        await self._admit_inference("other-model-tile", "model-b")

        claims = []
        while claim := await self.repository.claim_due("worker", timedelta(seconds=30)):
            claims.append(claim)

        operations = sorted((claim.operation.value, claim.model_id) for claim in claims)
        assert operations == [
            ("builder", "model-a"),
            *[("inference", "model-a")] * INFERENCE_LIMIT,
            ("inference", "model-b"),
        ], operations

    async def test_claims_are_limited_to_the_requested_operations(self) -> None:
        builder_id = await self._admit_execution("builder")
        await self._advance_time(timedelta(milliseconds=10))
        await self._admit_inference("tile")

        inference = await self.repository.claim_due(
            "worker", timedelta(seconds=30), [Operation.INFERENCE]
        )
        builder = await self.repository.claim_due(
            "worker", timedelta(seconds=30), [Operation.BUILDER]
        )

        assert inference is not None and inference.operation is Operation.INFERENCE
        assert builder is not None and builder.execution_id == builder_id
        assert (
            await self.repository.claim_due(
                "worker", timedelta(seconds=30), [Operation.BUILDER]
            )
            is None
        )

    async def test_cancelled_or_failed_inference_queues_no_report(self) -> None:
        cancelled_id = await self._admit_inference("cancelled")
        failed_id = await self._admit_inference("failed")
        await self.repository.request_cancel(cancelled_id)
        claim = await self.repository.claim_due("worker", timedelta(seconds=30))
        assert claim is not None and claim.execution_id == failed_id
        await self.repository.settle(
            failed_id,
            claim.fencing_token,
            ProcessingFailed(ExecutionError("model_rejected_input", "no"), False),
        )

        cancelled = await self.repository.get(cancelled_id)
        failed = await self.repository.get(failed_id)
        assert cancelled.status is ExecutionStatus.CANCELLED
        assert failed.status is ExecutionStatus.FAILED
        assert cancelled.work_kind is None and failed.work_kind is None
        assert await self.repository.claim_due("worker", timedelta(seconds=30)) is None
        for execution_id in (cancelled_id, failed_id):
            try:
                await self.repository.inference_content(execution_id)
            except ResultUnavailableError:
                pass
            else:
                raise AssertionError("unfinished inference returned content")

    async def test_finished_inference_is_deleted_after_retention(self) -> None:
        finished_id = await self._admit_inference("finished")
        builder_id = await self._admit_execution("builder")
        await self.repository.request_cancel(finished_id)
        await self.repository.request_cancel(builder_id)
        queued_id = await self._admit_inference("queued", "model-b")
        await self.repository.maintain()
        await self.repository.get(finished_id)

        await self._advance_time(INFERENCE_RETENTION * 2)
        await self.repository.maintain()

        try:
            await self.repository.get(finished_id)
        except UnknownExecutionError:
            pass
        else:
            raise AssertionError("finished inference outlived its retention")
        await self.repository.get(builder_id)
        await self.repository.get(queued_id)
        # The replay key is free again once the execution is gone.
        assert await self._admit_inference("finished") != finished_id

    async def test_get_many_keeps_order_and_skips_unknown_ids(self) -> None:
        first = await self._admit_inference("first")
        second = await self._admit_execution("second")

        found = await self.repository.get_many(
            [second, "not-an-id", "00000000-0000-0000-0000-000000000000", first]
        )

        assert [execution.id for execution in found] == [second, first]

    async def test_inference_claims_in_batches_within_model_limits(self) -> None:
        for index in range(INFERENCE_LIMIT + 1):
            await self._admit_inference(f"a-{index}", payload=f"a-{index}".encode())
        await self._admit_inference("b-0", "model-b", payload=b"b-0")
        await self._admit_execution("builder")

        claims = await self.repository.claim_inference(
            "worker", timedelta(seconds=30), INFERENCE_LIMIT + 5
        )

        assert sorted(claim.payload or b"" for claim in claims) == [
            *(f"a-{index}".encode() for index in range(INFERENCE_LIMIT)),
            b"b-0",
        ], claims
        assert all(claim.operation is Operation.INFERENCE for claim in claims)
        assert (
            await self.repository.claim_inference("worker", timedelta(seconds=30), 5)
            == []
        )
        limited = await self.repository.claim_inference(
            "worker", timedelta(seconds=30), 0
        )
        assert limited == []
