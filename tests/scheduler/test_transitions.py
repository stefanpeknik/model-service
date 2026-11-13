"""State-machine rules that hold whichever repository applies them."""

from __future__ import annotations

import unittest
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from model_service.contracts.invocation import InputAccessContext, Invocation
from model_service.contracts.result import BuilderResult
from model_service.scheduler.internal_state import Admission, RetryPolicy, WorkKind
from model_service.scheduler.public import SubmissionIdentity
from model_service.scheduler.status import ExecutionErrorCode, ExecutionStatus
from model_service.scheduler.transitions import ExecutionTransitions


NOW = datetime(2026, 11, 2, tzinfo=UTC)
POLICY = RetryPolicy(
    max_attempts=3,
    initial_backoff=timedelta(seconds=1),
    maximum_backoff=timedelta(seconds=1),
)
TRANSITIONS = ExecutionTransitions(POLICY, POLICY)


def publishing():
    deadline = NOW + timedelta(minutes=5)
    admitted = TRANSITIONS.admission.admit(
        Admission(
            identity=SubmissionIdentity("external", "job", "job"),
            builder_id="heatmap",
            model_id="model",
            invocation=Invocation(
                inputs={},
                resolved_configuration=None,
                input_access=InputAccessContext(kind="external", state={}),
                deadline_at=deadline,
            ),
            protected_context={},
            deadline_at=deadline,
        ),
        "execution",
        NOW,
    )
    return replace(
        admitted,
        status=ExecutionStatus.RUNNING,
        work_kind=WorkKind.PROVIDER_FINISH,
        result=BuilderResult({}),
        finish_attempts=3,
        fencing_token="token",
        lease_owner="worker",
        lease_expires_at=NOW + timedelta(hours=1),
    )


class TransitionTests(unittest.TestCase):
    def test_a_cancellation_report_gets_its_own_attempts(self) -> None:
        cancelled = TRANSITIONS.admission.cancel(publishing(), NOW)

        self.assertIs(cancelled.work_kind, WorkKind.PROVIDER_CANCELLATION_REPORT)
        self.assertEqual(cancelled.finish_attempts, 0)

    def test_an_unreadable_context_fails_without_a_report(self) -> None:
        failed = TRANSITIONS.outcomes.unreadable(publishing(), NOW)

        self.assertIs(failed.status, ExecutionStatus.FAILED)
        self.assertIsNone(failed.work_kind)
        assert failed.error is not None
        self.assertEqual(
            failed.error.code, ExecutionErrorCode.PROTECTED_CONTEXT_UNREADABLE
        )

    def test_an_unreadable_context_keeps_an_outcome_already_reached(self) -> None:
        cancelled = TRANSITIONS.admission.cancel(publishing(), NOW)

        ended = TRANSITIONS.outcomes.unreadable(cancelled, NOW)

        self.assertIs(ended.status, ExecutionStatus.CANCELLED)
        self.assertIsNone(ended.work_kind)
