"""The analysis runner's HTTP format: the run request and the error envelope, used by both ends."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated

from pydantic import Field, StrictBool, TypeAdapter

from model_service.contracts.invocation import ExecutionContext, Invocation
from model_service.contracts.json_values import JsonObject


@dataclass(frozen=True, slots=True)
class AnalysisRunRequest:
    """One run of a resolved analysis.

    Attributes:
        invocation: Inputs, their access, deadline and output path. Its
            `configuration` is only a record: the settings are already in
            `analysis`.
        analysis: The document `catalog.analyses.resolve_run` made at admission.
        execution_context: Attempt identity and protected provider state.
    """

    invocation: Invocation
    analysis: JsonObject
    execution_context: ExecutionContext = field(default_factory=ExecutionContext)


ANALYSIS_RUN_REQUEST: TypeAdapter[AnalysisRunRequest] = TypeAdapter(AnalysisRunRequest)


class ErrorCode(StrEnum):
    """Error codes the analysis runner answers with over HTTP."""

    # Builder-era names: callers and stored executions read these codes.
    INVALID_INVOCATION = "invalid_builder_invocation"
    UNAVAILABLE = "builder_unavailable"


@dataclass(frozen=True, slots=True)
class ErrorDetail:
    """An analysis runner's error.

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
    """The error body the analysis runner returns.

    Attributes:
        detail: The error.
    """

    detail: ErrorDetail
