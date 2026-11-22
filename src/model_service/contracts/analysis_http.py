"""The analysis runner's HTTP request, used by the API and the runner deployment.

Errors use the builder deployment's envelope (`contracts/builder_http.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import TypeAdapter

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
