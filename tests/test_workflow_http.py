from __future__ import annotations

import unittest
from datetime import UTC, datetime
from typing import Never, cast

from fastapi import HTTPException

from workflow_execution.contracts import (
    InputAccessContext,
    InvalidWorkflowInvocation,
    WorkflowContractViolation,
    WorkflowExecutionContext,
    WorkflowFailure,
    WorkflowInvocation,
)
from workflow_execution.http import WorkflowHTTPErrorCode, run_workflow_http


class FailingApplication:
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def run(
        self,
        _invocation: WorkflowInvocation,
        _execution_context: WorkflowExecutionContext | None = None,
    ) -> Never:
        raise self._error


class WorkflowHttpTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_invocation_is_a_permanent_client_error(self) -> None:
        error = await self._run(InvalidWorkflowInvocation("invalid input"))
        self.assertEqual(error.status_code, 422)
        detail = cast("dict[str, object]", error.detail)
        self.assertEqual(detail["code"], WorkflowHTTPErrorCode.INVALID_INVOCATION)
        self.assertFalse(detail["retryable"])

    async def test_contract_violation_hides_internal_details(self) -> None:
        error = await self._run(WorkflowContractViolation("private detail"))
        self.assertEqual(error.status_code, 500)
        detail = cast("dict[str, object]", error.detail)
        self.assertEqual(detail["code"], WorkflowHTTPErrorCode.CONTRACT_VIOLATION)
        self.assertNotIn("private detail", str(detail["message"]))

    async def test_workflow_failure_preserves_safe_classification(self) -> None:
        error = await self._run(
            WorkflowFailure(
                code="input_unavailable",
                safe_message="input is temporarily unavailable",
                retryable=True,
            )
        )
        self.assertEqual(error.status_code, 503)
        detail = cast("dict[str, object]", error.detail)
        self.assertEqual(detail["code"], "input_unavailable")
        self.assertEqual(detail["message"], "input is temporarily unavailable")
        self.assertTrue(detail["retryable"])

    async def _run(self, failure: Exception) -> HTTPException:
        invocation = WorkflowInvocation(
            inputs={},
            resolved_configuration={},
            input_access=InputAccessContext(kind="test", state={}),
            deadline_at=datetime.now(UTC),
        )
        with self.assertRaises(HTTPException) as raised:
            await run_workflow_http(FailingApplication(failure), invocation)
        return raised.exception


if __name__ == "__main__":
    unittest.main()
