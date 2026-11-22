"""Analysis files (`catalog/analyses/<id>.yaml`) and what they resolve to.

An analysis is one slide-in, results-out job: a source, steps (`{model: <id>}`
calls a served model, `{use: <step>}` a component) and sinks. Loading checks
the tile geometry against the first model step, folds the columns from source
to sinks with the models' dimensions, checks the frame fields sinks read and
collects the per-run settings. The analysis runner executes the result.
"""

from model_service.catalog.analyses.declaration import Analysis, AnalysisStep
from model_service.catalog.analyses.kind import FOLDER, Analyses
from model_service.catalog.analyses.model_step import ModelStep


__all__ = ["FOLDER", "Analyses", "Analysis", "AnalysisStep", "ModelStep"]
