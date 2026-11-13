"""The builder deployment's HTTP format: the run request and the error envelope, used by both ends."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated

from pydantic import Field, StrictBool

from model_service.contracts.invocation import ExecutionContext, Invocation
from model_service.contracts.model_declarations import ModelBinding


@dataclass(frozen=True, slots=True)
class RunRequest:
    """The stored invocation plus the model resolved for this attempt."""

    invocation: Invocation
    model: ModelBinding
    execution_context: ExecutionContext = field(default_factory=ExecutionContext)


class ErrorCode(StrEnum):
    INVALID_INVOCATION = "invalid_builder_invocation"
    UNAVAILABLE = "builder_unavailable"
    CONTRACT_VIOLATION = "builder_contract_violation"


@dataclass(frozen=True, slots=True)
class ErrorDetail:
    code: Annotated[str, Field(strict=True, min_length=1)]
    message: Annotated[str, Field(strict=True, min_length=1)]
    retryable: StrictBool


@dataclass(frozen=True, slots=True)
class ErrorResponse:
    detail: ErrorDetail
