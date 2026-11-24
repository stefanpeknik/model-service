"""Pipeline files (`catalog/pipelines/<id>.yaml`) and what they resolve to.

A pipeline is one slide-in, results-out job: a source, steps (`{model: <id>}`
calls a model, `{use: <step>}` a component) and sinks. Loading checks
the tile geometry against the first model step, folds the columns from source
to sinks with the models' dimensions, checks the frame fields sinks read and
collects the per-run settings. `resolve_run` turns one into the document a run
executes, and `load_resolved` reads that document back on the runner.
"""

from model_service.catalog.pipelines.declaration import Pipeline, PipelineStep
from model_service.catalog.pipelines.document import (
    FOLDER,
    build_pipeline,
    pipeline_schema,
)
from model_service.catalog.pipelines.model_step import ModelStep
from model_service.catalog.pipelines.resolved import (
    ResolvedPipeline,
    ResolvedPipelineInvalid,
    load_resolved,
    resolve_run,
)


__all__ = [
    "FOLDER",
    "ModelStep",
    "Pipeline",
    "PipelineStep",
    "ResolvedPipeline",
    "ResolvedPipelineInvalid",
    "build_pipeline",
    "load_resolved",
    "pipeline_schema",
    "resolve_run",
]
