"""Check safe HTTP translation of builder failures."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Never, cast

from fastapi import HTTPException

from model_service.builders.contracts import (
    BuilderFailure,
    ContractViolation,
    ExecutionContext,
    InputAccessContext,
    InvalidInvocation,
    Invocation,
    SpatialScoresModel,
)
from model_service.builders.http import ErrorCode, RunRequest, run_http


if TYPE_CHECKING:
    from model_service.builders.application import BuilderApplication
    from model_service.builders.contracts import ModelBinding


class FailingApplication:
    def __init__(self, error: Exception) -> None:
        self._error = error

    async def run(
        self,
        _invocation: Invocation,
        _model: ModelBinding,
        _execution_context: ExecutionContext | None = None,
    ) -> Never:
        raise self._error


class BuilderHttpTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_invocation_is_a_permanent_client_error(self) -> None:
        detail, status_code = await self._run(InvalidInvocation("missing slide"))

        self.assertEqual(status_code, 422)
        self.assertEqual(detail["code"], ErrorCode.INVALID_INVOCATION)
        self.assertEqual(detail["message"], "missing slide")
        self.assertFalse(detail["retryable"])

    async def test_contract_violation_hides_internal_details(self) -> None:
        detail, status_code = await self._run(ContractViolation("private detail"))

        self.assertEqual(status_code, 500)
        self.assertEqual(detail["code"], ErrorCode.CONTRACT_VIOLATION)
        self.assertNotIn("private detail", str(detail["message"]))

    async def test_builder_failure_keeps_its_safe_classification(self) -> None:
        for retryable, status_code in ((True, 503), (False, 422)):
            with self.subTest(retryable=retryable):
                detail, actual = await self._run(
                    BuilderFailure(
                        code="model_unavailable",
                        safe_message="model is unavailable",
                        retryable=retryable,
                    )
                )
                self.assertEqual(actual, status_code)
                self.assertEqual(
                    detail,
                    {
                        "code": "model_unavailable",
                        "message": "model is unavailable",
                        "retryable": retryable,
                    },
                )

    async def _run(self, failure: Exception) -> tuple[dict[str, object], int]:
        request = RunRequest(
            invocation=Invocation(
                inputs={},
                resolved_configuration=None,
                input_access=InputAccessContext(kind="test", state={}),
                deadline_at=datetime.now(UTC),
            ),
            model=SpatialScoresModel(
                id="model", tile_size=8, output_tile_size=1, n_channels=1, mpp=1.0
            ),
        )
        application = cast("BuilderApplication", FailingApplication(failure))
        with self.assertRaises(HTTPException) as raised:
            await run_http(application, request)
        return cast("dict[str, object]", raised.exception.detail), (
            raised.exception.status_code
        )


if __name__ == "__main__":
    unittest.main()
