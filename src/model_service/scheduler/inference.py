"""Run one raw model request as an execution.

The facade implements `InferenceTarget` over the model catalog and Ray Serve.
Admission and every attempt check the payload against the model's current
input, so a queued tile is never reinterpreted after a model change.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

from model_service.scheduler.contracts import (
    INFERENCE_PROVIDER,
    ExecutionError,
    ExecutionErrorCode,
    InferenceResult,
)
from model_service.scheduler.domain import (
    InferenceSucceeded,
    ProcessingFailed,
    ProcessingRejected,
    SubmissionIdentity,
)


if TYPE_CHECKING:
    from collections.abc import Mapping

    from model_service.scheduler.contracts import InferenceScheduleRequest
    from model_service.scheduler.domain import Settlement


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


class InferenceTarget(ABC):
    @abstractmethod
    def check(self, model_id: str, payload: bytes) -> None:
        """Check that a model exists and accepts the payload under its current input.

        Args:
            model_id: Catalog ID of the model.
            payload: Raw request body.

        Raises:
            UnknownModelError: The model is not in the catalog.
            IncompatibleInputError: The payload does not fit the model's current input.
        """
        ...

    @abstractmethod
    async def call(
        self, model_id: str, payload: bytes, headers: Mapping[str, str]
    ) -> ModelResponse:
        """Send the payload to the model and return its HTTP response.

        Args:
            model_id: Catalog ID of the model.
            payload: Raw request body.
            headers: Caller headers to forward with the request.

        Returns:
            The model's status, body and headers.

        Raises:
            ModelUnavailableError: The request got no HTTP response.
            ResultTooLargeError: The response body exceeds the size limit.
        """
        ...


def identify(request: InferenceScheduleRequest) -> SubmissionIdentity:
    """Compute the replay identity of an inference request.

    The fingerprint hashes the model ID, sorted headers, requested timeout and the
    payload's SHA-256, so a reused idempotency key with different content is detected.

    Args:
        request: Inference request to identify.

    Returns:
        The identity scoped to the inference provider and the request's idempotency key.
    """
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
        provider=INFERENCE_PROVIDER,
        key=request.idempotency_key,
        fingerprint=hashlib.sha256(canonical).hexdigest(),
    )


_REJECTED_INPUT = frozenset({400, 413, 415, 422})
_BACKPRESSURE = frozenset({429, 503})


def settlement(response: ModelResponse) -> Settlement:
    """Map the model's HTTP response to an attempt outcome.

    2xx succeeds with the body and its `x-` headers (except `x-request-id`); 429
    and 503 are backpressure honouring `Retry-After`; 400, 413, 415 and 422 fail
    permanently as rejected input; any other status fails retryably.

    Args:
        response: The model's HTTP response.

    Returns:
        The settlement for the attempt.
    """
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
    """Parse a `Retry-After` header given in seconds.

    Args:
        value: Header value, or None when absent.

    Returns:
        The delay, or None when absent, not a number, or outside 0 to 3600 seconds.
    """
    try:
        seconds = float(value) if value is not None else None
    except ValueError:
        return None
    if seconds is None or not 0 <= seconds <= 3600:
        return None
    return timedelta(seconds=seconds)
