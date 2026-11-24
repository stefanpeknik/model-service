"""The pipeline runner: executes any resolved pipeline on one set of inputs.

`run_pipeline(pipeline, inputs, outputs, models)` is a plain async function,
with no Ray: it owns concurrency, cancellation, progress (the share of the
source's keys done), per-step timing, failure codes, ordered delivery when a
sink needs it, and all-or-nothing writes across sinks. Model steps call each
model's handle through the host's `infer(tile)`; tests pass a fake
model with the same `infer`.
"""

from model_service.pipeline_runner.models import ModelHandle
from model_service.pipeline_runner.run import Concurrency, run_pipeline


__all__ = ["Concurrency", "ModelHandle", "run_pipeline"]
