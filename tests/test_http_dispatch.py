"""Check workflow HTTP result parsing and rejection/failure/unknown-outcome classification."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from workflow_execution.contracts import (
    InlineOutput,
    InputAccessContext,
    WorkflowExecutionContext,
    WorkflowInvocation,
    WorkflowResult,
)
from workflow_execution.dispatch import (
    ConfirmedDispatchRejectionError,
    HTTPWorkflowDispatch,
    RemoteWorkflowFailureError,
    UnknownDispatchOutcomeError,
)


if TYPE_CHECKING:
    from collections.abc import Mapping

    from workflow_execution.dispatch import HTTPResponse


class FakeResponse:
    """Provide a configurable workflow HTTP response without a network connection."""

    def __init__(
        self,
        status_code: int,
        body: object,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Configure response status, decoded body, and optional headers for dispatch classification.

        Args:
            status_code: HTTP status to return or classify.
            body: Typed or JSON request payload accepted by this boundary.
            headers: HTTP metadata to filter, send, or include in an error response.
        """
        self.status_code = status_code
        self._body = body
        self.headers = headers or {}

    def json(self) -> object:
        """Return the predetermined decoded response body.

        Returns:
            object: The delegated operation's result or controlled fixture value
                described above.
        """
        return self._body


class FakeHttpClient:
    """Record workflow POST arguments and return a predetermined fake response."""

    def __init__(self, response: FakeResponse) -> None:
        """Retain the response to return and initialize an empty request log.

        Args:
            response: Upstream HTTP response or mutable FastAPI response used to select
                the public status.
        """
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
        """Record URL, body, headers, and timeout and return the configured response.

        Args:
            url: Complete upstream HTTP request URL.
            json: JSON-compatible workflow request passed to the HTTP client.
            headers: HTTP metadata to filter, send, or include in an error response.
            timeout: Request or scenario time budget in seconds.

        Returns:
            HTTPResponse: HTTP response used to classify workflow dispatch acceptance
                and outcome.
        """
        self.requests.append(
            {"url": url, "json": json, "headers": headers, "timeout": timeout}
        )
        return self.response


class HttpWorkflowDispatchTests(unittest.IsolatedAsyncioTestCase):
    """Exercise workflow HTTP result parsing and rejection/failure/unknown-outcome classification."""

    async def test_ray_backpressure_uses_configured_429_signal(self) -> None:
        """Verify Ray backpressure uses configured 429 signal.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
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
        """Verify generic 503 is an unknown outcome.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
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
        """Verify classified workflow failure keeps safe error.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
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
        """Verify success parses normalized result.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
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
        """Build a typed workflow invocation with the fixture access context and a bounded future deadline.

        Returns:
            WorkflowInvocation: Typed inputs, resolved configuration, access settings,
                and absolute deadline.
        """
        return WorkflowInvocation(
            inputs={},
            resolved_configuration={},
            input_access=InputAccessContext(kind="shared_storage", state={}),
            deadline_at=datetime.now(UTC) + timedelta(minutes=1),
        )


if __name__ == "__main__":
    unittest.main()
