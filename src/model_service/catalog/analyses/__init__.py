"""Analysis files (`catalog/analyses/<id>.yaml`) and what they resolve to.

An analysis is one slide-in, results-out job: a source, steps (`{model: <id>}`
calls a served model, `{use: <step>}` a component) and sinks. Loading checks
the tile geometry against the first model step, folds the columns from source
to sinks with the models' dimensions, checks the frame fields sinks read and
collects the per-run settings. `resolve_run` turns one into the document a run
executes, and `load_resolved` reads that document back on the runner.
"""

from model_service.catalog.analyses.declaration import Analysis, AnalysisStep
from model_service.catalog.analyses.kind import FOLDER, Analyses
from model_service.catalog.analyses.model_step import ModelStep
from model_service.catalog.analyses.resolved import (
    ResolvedAnalysis,
    ResolvedAnalysisInvalid,
    load_resolved,
    resolve_run,
)


__all__ = [
    "FOLDER",
    "Analyses",
    "Analysis",
    "AnalysisStep",
    "ModelStep",
    "ResolvedAnalysis",
    "ResolvedAnalysisInvalid",
    "load_resolved",
    "resolve_run",
]
