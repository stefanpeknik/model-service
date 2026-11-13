"""The builder deployment's HTTP format: the run request and the error envelope, used by both ends."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated

from pydantic import Field, StrictBool, TypeAdapter

from model_service.contracts.invocation import ExecutionContext, Invocation
from model_service.contracts.model_declarations import ModelBinding


@dataclass(frozen=True, slots=True)
class RunRequest:
    """The stored invocation plus the model resolved for this attempt."""

    invocation: Invocation
    model: ModelBinding
    execution_context: ExecutionContext = field(default_factory=ExecutionContext)


RUN_REQUEST: TypeAdapter[RunRequest] = TypeAdapter(RunRequest)


class ErrorCode(StrEnum):
    """Error codes a builder replica answers with over HTTP."""

    INVALID_INVOCATION = "invalid_builder_invocation"
    UNAVAILABLE = "builder_unavailable"
    CONTRACT_VIOLATION = "builder_contract_violation"


@dataclass(frozen=True, slots=True)
class ErrorDetail:
    """A builder replica's error.

    Attributes:
        code: Machine-readable error code.
        message: Message safe to show callers.
        retryable: Whether the same invocation may succeed later.
    """

    code: Annotated[str, Field(strict=True, min_length=1)]
    message: Annotated[str, Field(strict=True, min_length=1)]
    retryable: StrictBool


@dataclass(frozen=True, slots=True)
class ErrorResponse:
    """The error body a builder replica returns.

    Attributes:
        detail: The error.
    """

    detail: ErrorDetail
