# Built-in workflows

This package contains concrete workflow implementations and their registration.
The reusable invocation runtime is in `../workflow_execution/`.
Declarations remain in `../workflow-definitions/` so the runtime catalog and local
EAD exporter consume the same contracts.

## Start here

- `registry.py` composes the built-in declarations, configuration validators,
  and defaults. `WorkflowCatalog.default()` loads this registration lazily.
- `heatmap/prostate.py` owns the prostate workflow, its configuration schema,
  defaults, atomic output completion, and normalized artifact result.
- `heatmap/processor.py` owns slide tiling, bounded prediction concurrency,
  reconstruction, mask aggregation, and TIFF writing.
- `heatmap/ray_predictor.py` supplies model configuration and predictions through
  a Ray model handle.
- `heatmap/tile_reader.py` reads tissue-selected tiles for the processor.
- `heatmap/prostate_serve.py` wires the runtime, processor, and input adapters
  into the Ray Serve application at `workflows.heatmap.prostate_serve:app`.

The legacy `../builders/heatmap_builder.py` entry point uses the same processor
and Ray predictor. Its model selection remains caller-supplied. The prostate
workflow's model selection belongs to its Serve wiring, not to the declaration
or EMPAIA configuration.

Workflow processing receives provider-neutral readers and returns outputs.
Provider configuration fetches, publication, and finalization remain in
`../integrations/`; scheduling and execution state remain in
`../execution_scheduler/`.
