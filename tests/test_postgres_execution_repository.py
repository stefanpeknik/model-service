from __future__ import annotations

import asyncio
import base64
import os
import unittest
from datetime import UTC, datetime, timedelta
from uuid import uuid4

from psycopg import AsyncConnection, sql

from execution_scheduler.contracts import ScheduleConflictError
from execution_scheduler.domain import (
    Admission,
    ProcessingSucceeded,
    ProviderFinishSucceeded,
    RenewDirective,
    RetryPolicy,
    StaleWorkClaimError,
    SubmissionIdentity,
    WorkKind,
)
from execution_scheduler.postgres import PostgresExecutionRepository
from execution_scheduler.security import ProtectedContextCipher
from workflow_execution.contracts import (
    InlineOutput,
    InputAccessContext,
    WorkflowInvocation,
    WorkflowResult,
)


DATABASE_DEADLINE_WINDOW = timedelta(milliseconds=500)
DATABASE_DEADLINE_WAIT_SECONDS = 0.7


class PostgresExecutionRepositoryTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        dsn = os.environ.get("TEST_POSTGRES_DSN")
        if dsn is None:
            self.skipTest("TEST_POSTGRES_DSN is not configured")
        self.dsn = dsn
        self.schema = f"test_execution_{uuid4().hex}"
        key = base64.urlsafe_b64encode(bytes(range(32))).decode()
        self.cipher = ProtectedContextCipher.from_base64(key)
        self.processing_policy = RetryPolicy(
            max_attempts=3,
            initial_backoff=timedelta(0),
            maximum_backoff=timedelta(0),
        )
        self.repository = self._repository()
        await self.repository.initialize()

    async def asyncTearDown(self) -> None:
        if not hasattr(self, "dsn"):
            return
        async with await AsyncConnection.connect(self.dsn) as connection:
            await connection.execute(
                sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(self.schema))
            )

    async def test_concurrent_equivalent_admission_creates_one_execution(self) -> None:
        admission = self._admission("submission-1")

        first, second = await asyncio.gather(
            self.repository.admit(admission),
            self._repository().admit(admission),
        )

        self.assertEqual({first.created, second.created}, {False, True})
        self.assertEqual(first.execution.id, second.execution.id)

    async def test_conflicting_admission_is_rejected(self) -> None:
        first = self._admission("submission-1")
        conflicting = Admission(
            identity=SubmissionIdentity(
                provider=first.identity.provider,
                key=first.identity.key,
                fingerprint="different-request",
            ),
            workflow_id=first.workflow_id,
            invocation=first.invocation,
            protected_context=first.protected_context,
            deadline_at=first.deadline_at,
        )
        await self.repository.admit(first)

        with self.assertRaises(ScheduleConflictError):
            await self._repository().admit(conflicting)

    async def test_workflow_limit_and_fencing_hold_across_connections(self) -> None:
        first = await self.repository.admit(self._admission("submission-1"))
        second = await self.repository.admit(self._admission("submission-2"))

        first_claim, competing_claim = await asyncio.gather(
            self.repository.claim_due("worker-1", timedelta(seconds=30)),
            self._repository().claim_due("worker-2", timedelta(seconds=30)),
        )

        claims = [
            claim for claim in (first_claim, competing_claim) if claim is not None
        ]
        self.assertEqual(len(claims), 1)
        claimed = claims[0]
        self.assertIn(
            claimed.execution_id,
            {first.execution.id, second.execution.id},
        )

    async def test_expired_processing_lease_is_recovered_and_stale_settlement_fails(
        self,
    ) -> None:
        admitted = await self.repository.admit(self._admission("submission-1"))
        abandoned = await self.repository.claim_due(
            "worker-1",
            timedelta(milliseconds=30),
        )
        assert abandoned is not None

        await asyncio.sleep(0.08)
        replacement = await self._repository().claim_due(
            "worker-2",
            timedelta(seconds=30),
        )

        assert replacement is not None
        self.assertEqual(replacement.execution_id, admitted.execution.id)
        self.assertNotEqual(replacement.fencing_token, abandoned.fencing_token)
        self.assertEqual(replacement.attempt, 2)
        result = WorkflowResult(outputs={"score": InlineOutput(value=0.9)})
        with self.assertRaises(StaleWorkClaimError):
            await self.repository.settle(
                abandoned.execution_id,
                abandoned.fencing_token,
                ProcessingSucceeded(result),
            )
        processing_complete = await self.repository.settle(
            replacement.execution_id,
            replacement.fencing_token,
            ProcessingSucceeded(result),
        )
        self.assertEqual(processing_complete.status, "running")
        self.assertEqual(processing_complete.work_kind, WorkKind.PROVIDER_FINISH)
        self.assertEqual(processing_complete.result, result)

        finish = await self.repository.claim_due(
            "worker-2",
            timedelta(seconds=30),
        )
        assert finish is not None
        succeeded = await self.repository.settle(
            finish.execution_id,
            finish.fencing_token,
            ProviderFinishSucceeded(),
        )
        self.assertEqual(succeeded.status, "succeeded")
        self.assertEqual(succeeded.result, result)

    async def test_renewal_and_cancellation_fence_competing_workers(self) -> None:
        admitted = await self.repository.admit(self._admission("submission-1"))
        claim = await self.repository.claim_due("worker-1", timedelta(seconds=1))
        assert claim is not None

        directive = await self.repository.renew(
            claim.execution_id,
            claim.fencing_token,
            timedelta(seconds=1),
        )
        competing = await self._repository().claim_due(
            "worker-2",
            timedelta(seconds=1),
        )

        self.assertEqual(directive, RenewDirective.CONTINUE)
        self.assertIsNone(competing)
        cancelled = await self._repository().request_cancel(admitted.execution.id)
        self.assertEqual(cancelled.status, "cancelled")
        self.assertEqual(
            await self.repository.renew(
                claim.execution_id,
                claim.fencing_token,
                timedelta(seconds=1),
            ),
            RenewDirective.CANCEL,
        )
        with self.assertRaises(StaleWorkClaimError):
            await self.repository.settle(
                claim.execution_id,
                claim.fencing_token,
                ProcessingSucceeded(WorkflowResult(outputs={})),
            )

    async def test_expired_finish_lease_preserves_result_and_processing_attempt(
        self,
    ) -> None:
        admitted = await self.repository.admit(self._admission("submission-1"))
        processing = await self.repository.claim_due(
            "worker-1",
            timedelta(seconds=1),
        )
        assert processing is not None
        result = WorkflowResult(outputs={"score": InlineOutput(value=0.9)})
        await self.repository.settle(
            processing.execution_id,
            processing.fencing_token,
            ProcessingSucceeded(result),
        )
        abandoned_finish = await self.repository.claim_due(
            "worker-1",
            timedelta(milliseconds=30),
        )
        assert abandoned_finish is not None

        await asyncio.sleep(0.08)
        replacement = await self._repository().claim_due(
            "worker-2",
            timedelta(seconds=1),
        )

        assert replacement is not None
        self.assertEqual(replacement.execution_id, admitted.execution.id)
        self.assertEqual(replacement.kind, WorkKind.PROVIDER_FINISH)
        self.assertEqual(replacement.result, result)
        self.assertEqual(replacement.attempt, 2)
        durable = await self.repository.get(admitted.execution.id)
        self.assertEqual(durable.processing_attempts, 1)

    async def test_database_deadline_rejects_a_late_settlement(self) -> None:
        admission = self._admission("submission-1")
        expiring = Admission(
            identity=admission.identity,
            workflow_id=admission.workflow_id,
            invocation=admission.invocation,
            protected_context=admission.protected_context,
            deadline_at=datetime.now(UTC) + DATABASE_DEADLINE_WINDOW,
        )
        admitted = await self.repository.admit(expiring)
        claim = await self.repository.claim_due("worker-1", timedelta(seconds=1))
        assert claim is not None

        await asyncio.sleep(DATABASE_DEADLINE_WAIT_SECONDS)
        with self.assertRaises(StaleWorkClaimError):
            await self.repository.settle(
                claim.execution_id,
                claim.fencing_token,
                ProcessingSucceeded(WorkflowResult(outputs={})),
            )
        failed = await self.repository.get(admitted.execution.id)
        self.assertEqual(failed.status, "failed")
        assert failed.error is not None
        self.assertEqual(failed.error.code, "execution_deadline_exceeded")

    async def test_deadline_failure_remains_reportable_until_provider_deadline(
        self,
    ) -> None:
        admission = self._admission("submission-report-deadline")
        execution_deadline = datetime.now(UTC) + DATABASE_DEADLINE_WINDOW
        expiring = Admission(
            identity=admission.identity,
            workflow_id=admission.workflow_id,
            invocation=admission.invocation,
            protected_context=admission.protected_context,
            deadline_at=execution_deadline,
            terminal_report_deadline_at=datetime.now(UTC) + timedelta(minutes=5),
        )
        await self.repository.admit(expiring)
        processing = await self.repository.claim_due(
            "worker-1",
            timedelta(seconds=1),
        )
        assert processing is not None
        await asyncio.sleep(DATABASE_DEADLINE_WAIT_SECONDS)

        self.assertEqual(
            await self.repository.renew(
                processing.execution_id,
                processing.fencing_token,
                timedelta(seconds=1),
            ),
            RenewDirective.DEADLINE_EXCEEDED,
        )
        report = await self.repository.claim_due(
            "worker-2",
            timedelta(seconds=1),
        )

        assert report is not None
        self.assertEqual(report.kind, WorkKind.PROVIDER_FAILURE_REPORT)
        assert report.error is not None
        self.assertEqual(report.error.code, "execution_deadline_exceeded")

    async def test_stale_renewal_cannot_apply_deadline_transition(self) -> None:
        admission = self._admission("submission-1")
        expiring = Admission(
            identity=admission.identity,
            workflow_id=admission.workflow_id,
            invocation=admission.invocation,
            protected_context=admission.protected_context,
            deadline_at=datetime.now(UTC) + DATABASE_DEADLINE_WINDOW,
        )
        admitted = await self.repository.admit(expiring)
        claim = await self.repository.claim_due("worker-1", timedelta(seconds=5))
        assert claim is not None
        await asyncio.sleep(DATABASE_DEADLINE_WAIT_SECONDS)

        stale = await self._repository().renew(
            claim.execution_id,
            str(uuid4()),
            timedelta(seconds=1),
        )
        unchanged = await self.repository.get(admitted.execution.id)
        current = await self.repository.renew(
            claim.execution_id,
            claim.fencing_token,
            timedelta(seconds=1),
        )

        self.assertEqual(stale, RenewDirective.LEASE_LOST)
        self.assertEqual(unchanged.status, "running")
        self.assertEqual(current, RenewDirective.DEADLINE_EXCEEDED)
        self.assertEqual(
            (await self.repository.get(admitted.execution.id)).status,
            "failed",
        )

    async def test_protected_context_is_not_stored_as_plaintext(self) -> None:
        admission = self._admission("submission-1")
        admission = Admission(
            identity=admission.identity,
            workflow_id=admission.workflow_id,
            invocation=admission.invocation,
            protected_context={"credential": "secret-job-token"},
            deadline_at=admission.deadline_at,
        )
        admitted = await self.repository.admit(admission)

        async with await AsyncConnection.connect(self.dsn) as connection:
            cursor = await connection.execute(
                sql.SQL(
                    "SELECT encode(protected_context, 'escape') FROM {} WHERE id = %s"
                ).format(sql.Identifier(self.schema, "executions")),
                (admitted.execution.id,),
            )
            row = await cursor.fetchone()

        assert row is not None
        self.assertNotIn("secret-job-token", row[0])
        loaded = await self.repository.get(admitted.execution.id)
        self.assertEqual(
            loaded.protected_context,
            {"credential": "secret-job-token"},
        )

    def _repository(self) -> PostgresExecutionRepository:
        return PostgresExecutionRepository(
            self.dsn,
            cipher=self.cipher,
            schema=self.schema,
            per_workflow_limit=1,
            processing_policy=self.processing_policy,
            finish_policy=self.processing_policy,
        )

    @staticmethod
    def _admission(submission_key: str) -> Admission:
        deadline = datetime.now(UTC) + timedelta(minutes=5)
        return Admission(
            identity=SubmissionIdentity(
                provider="standalone",
                key=submission_key,
                fingerprint=f"fingerprint-{submission_key}",
            ),
            workflow_id="prostate-heatmap",
            invocation=WorkflowInvocation(
                inputs={},
                resolved_configuration={},
                input_access=InputAccessContext(
                    kind="shared_storage",
                    state={"root": "/shared"},
                ),
                deadline_at=deadline,
            ),
            protected_context={},
            deadline_at=deadline,
        )


if __name__ == "__main__":
    unittest.main()
