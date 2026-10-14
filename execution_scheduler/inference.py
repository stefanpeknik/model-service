"""Run one raw model request as an execution.

The facade implements `InferenceTarget` over the model catalog and Ray Serve.
Admission and every attempt check the payload against the model's current
input, so a queued tile is never reinterpreted after a model change.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING, Protocol

from execution_scheduler.contracts import (
    ExecutionError,
    ExecutionErrorCode,
    InferenceResult,
)
from execution_scheduler.domain import (
    InferenceSucceeded,
    ProcessingFailed,
    ProcessingRejected,
    SubmissionIdentity,
)


if TYPE_CHECKING:
    from collections.abc import Mapping

    from execution_scheduler.contracts import InferenceScheduleRequest
    from execution_scheduler.domain import Settlement


PROVIDER = "inference"


@dataclass(frozen=True, slots=True)
class ModelResponse:
    status_code: int
    content: bytes
    headers: dict[str, str]


class IncompatibleInputError(ValueError):
    """The payload does not fit the model's current input."""


class ModelUnavailableError(RuntimeError):
    """The request did not get an HTTP response from the model."""


class ResultTooLargeError(RuntimeError):
    pass


class InferenceTarget(Protocol):
    def check(self, model_id: str, payload: bytes) -> None:
        """Raise UnknownModelError or IncompatibleInputError."""
        ...

    async def call(
        self, model_id: str, payload: bytes, headers: Mapping[str, str]
    ) -> ModelResponse:
        """Raise ModelUnavailableError or ResultTooLargeError."""
        ...


def identify(request: InferenceScheduleRequest) -> SubmissionIdentity:
    canonical = json.dumps(
        {
            "model_id": request.model_id,
            "headers": sorted(request.headers.items()),
            "requested_timeout_seconds": request.requested_timeout_seconds,
            "payload_sha256": hashlib.sha256(request.payload).hexdigest(),
        },
        separators=(",", ":"),
    ).encode()
    return SubmissionIdentity(
        provider=PROVIDER,
        key=request.idempotency_key,
        fingerprint=hashlib.sha256(canonical).hexdigest(),
    )


_REJECTED_INPUT = frozenset({400, 413, 415, 422})
_BACKPRESSURE = frozenset({429, 503})


def settlement(response: ModelResponse) -> Settlement:
    """Map the model's HTTP response to an attempt outcome."""
    status = response.status_code
    if 200 <= status < 300:
        return InferenceSucceeded(
            InferenceResult(
                media_type=response.headers.get("content-type"),
                headers={
                    name: value
                    for name, value in response.headers.items()
                    if name.startswith("x-") and name != "x-request-id"
                },
                size=len(response.content),
            ),
            response.content,
        )
    if status in _BACKPRESSURE:
        return ProcessingRejected(_retry_after(response.headers.get("retry-after")))
    if status in _REJECTED_INPUT:
        return ProcessingFailed(
            ExecutionError(
                ExecutionErrorCode.MODEL_REJECTED_INPUT,
                f"model rejected the request with HTTP {status}",
            ),
            retryable=False,
        )
    return ProcessingFailed(
        ExecutionError(
            ExecutionErrorCode.MODEL_FAILED, f"model returned HTTP {status}"
        ),
        retryable=True,
    )


def _retry_after(value: str | None) -> timedelta | None:
    try:
        seconds = float(value) if value is not None else None
    except ValueError:
        return None
    if seconds is None or not 0 <= seconds <= 3600:
        return None
    return timedelta(seconds=seconds)
