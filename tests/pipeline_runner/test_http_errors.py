"""The runner answers failures in the envelope the control plane's dispatch reads."""

from __future__ import annotations

import asyncio
from typing import Never, cast

import pytest
from fastapi import HTTPException

from model_service.contracts.failures import InvalidInvocation, RunFailure
from model_service.contracts.pipeline_http import ErrorCode
from model_service.pipeline_runner.http_errors import answer_http


def answer(failure: Exception) -> tuple[dict[str, object], int]:
    async def run() -> Never:
        raise failure

    with pytest.raises(HTTPException) as raised:
        asyncio.run(answer_http(run()))
    return cast("dict[str, object]", raised.value.detail), raised.value.status_code


def test_an_invalid_invocation_is_a_permanent_client_error() -> None:
    detail, status_code = answer(InvalidInvocation("missing slide"))

    assert status_code == 422
    assert detail == {
        "code": ErrorCode.INVALID_INVOCATION,
        "message": "missing slide",
        "retryable": False,
    }


@pytest.mark.parametrize(("retryable", "status_code"), [(True, 503), (False, 422)])
def test_a_failure_keeps_its_safe_classification(
    retryable: bool, status_code: int
) -> None:
    detail, actual = answer(
        RunFailure(
            code="model_failed",
            safe_message="model is unavailable",
            retryable=retryable,
        )
    )

    assert actual == status_code
    assert detail == {
        "code": "model_failed",
        "message": "model is unavailable",
        "retryable": retryable,
    }
