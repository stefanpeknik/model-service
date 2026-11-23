"""The analysis runner: executes any resolved analysis on one set of inputs.

`run_analysis(analysis, inputs, outputs, models)` is a plain async function,
with no Ray: it owns concurrency, cancellation, progress (the share of the
source's keys done), per-step timing, failure codes, ordered delivery when a
sink needs it, and all-or-nothing writes across sinks. Model steps call each
served model's handle through the host's `infer(tile)`; tests pass
`components.testing.fake_model()`.
"""

from model_service.analysis_runner.models import ModelHandle
from model_service.analysis_runner.run import Concurrency, run_analysis


__all__ = ["Concurrency", "ModelHandle", "run_analysis"]
