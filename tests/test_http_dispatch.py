from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from execution_scheduler.dispatch import (
    ConfirmedDispatchRejectionError,
    HTTPWorkflowDispatch,
    RemoteWorkflowFailureError,
    UnknownDispatchOutcomeError,
)
from workflow_execution.contracts import (
    InlineOutput,
    InputAccessContext,
    WorkflowExecutionContext,
    WorkflowInvocation,
    WorkflowResult,
)


if TYPE_CHECKING:
    from collections.abc import Mapping

    from execution_scheduler.dispatch import HTTPResponse


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        body: object,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}

    def json(self) -> object:
        return self._body


class FakeHttpClient:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.requests: list[dict[str, object]] = []

    async def post(
        self,
        url: str,
        *,
        json: object,
        headers: Mapping[str, str],
        timeout: float,
    ) -> HTTPResponse:
        self.requests.append(
            {"url": url, "json": json, "headers": headers, "timeout": timeout}
        )
        return self.response


class HttpWorkflowDispatchTests(unittest.IsolatedAsyncioTestCase):
    async def test_ray_backpressure_uses_configured_429_signal(self) -> None:
        client = FakeHttpClient(
            FakeResponse(429, "queue full", headers={"retry-after": "7"})
        )
        dispatch = HTTPWorkflowDispatch(client, origin="http://ray-serve:8000")

        with self.assertRaises(ConfirmedDispatchRejectionError) as raised:
            await dispatch.run(
                "prostate-heatmap",
                WorkflowExecutionContext(processing_attempt_id="attempt-1"),
                self._invocation(),
            )

        self.assertEqual(raised.exception.retry_after, timedelta(seconds=7))

    async def test_generic_503_is_an_unknown_outcome(self) -> None:
        dispatch = HTTPWorkflowDispatch(
            FakeHttpClient(FakeResponse(503, "proxy unavailable")),
            origin="http://ray-serve:8000",
        )

        with self.assertRaises(UnknownDispatchOutcomeError):
            await dispatch.run(
                "prostate-heatmap",
                WorkflowExecutionContext(processing_attempt_id="attempt-1"),
                self._invocation(),
            )

    async def test_classified_workflow_failure_keeps_safe_error(self) -> None:
        dispatch = HTTPWorkflowDispatch(
            FakeHttpClient(
                FakeResponse(
                    503,
                    {
                        "detail": {
                            "code": "storage_temporarily_unavailable",
                            "message": "shared storage is unavailable",
                            "retryable": True,
                        }
                    },
                )
            ),
            origin="http://ray-serve:8000",
        )

        with self.assertRaises(RemoteWorkflowFailureError) as raised:
            await dispatch.run(
                "prostate-heatmap",
                WorkflowExecutionContext(processing_attempt_id="attempt-1"),
                self._invocation(),
            )

        self.assertEqual(raised.exception.code, "storage_temporarily_unavailable")
        self.assertTrue(raised.exception.retryable)

    async def test_success_parses_normalized_result(self) -> None:
        client = FakeHttpClient(
            FakeResponse(
                200,
                {"outputs": {"score": {"kind": "inline", "value": 0.9}}},
            )
        )
        dispatch = HTTPWorkflowDispatch(client, origin="http://ray-serve:8000")

        result = await dispatch.run(
            "prostate-heatmap",
            WorkflowExecutionContext(processing_attempt_id="attempt-1"),
            self._invocation(),
        )

        self.assertEqual(
            result,
            WorkflowResult(outputs={"score": InlineOutput(value=0.9)}),
        )
        self.assertEqual(
            client.requests[0]["url"],
            "http://ray-serve:8000/workflows/prostate-heatmap/",
        )
        self.assertEqual(
            client.requests[0]["headers"],
            {"x-processing-attempt-id": "attempt-1"},
        )

    @staticmethod
    def _invocation() -> WorkflowInvocation:
        return WorkflowInvocation(
            inputs={},
            resolved_configuration={},
            input_access=InputAccessContext(kind="shared_storage", state={}),
            deadline_at=datetime.now(UTC) + timedelta(minutes=1),
        )


if __name__ == "__main__":
    unittest.main()
