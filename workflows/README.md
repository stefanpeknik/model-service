# Built-in workflows

This package contains concrete workflow implementations and their registration.
The reusable invocation runtime is in `../workflow_execution/`.
Declarations remain in `../workflow-definitions/` so the runtime catalog and local
EAD exporter consume the same contracts.

## Start here

- `registry.py` composes the built-in declarations, configuration validators,
  defaults, and worker construction. `WorkflowCatalog.default()` loads this
  registration lazily. Importing the catalog does not import native image libraries.
- `segmentation/episeg.py` produces a floating-point score TIFF using the shared
  spatial processor and the `episeg-1` model handle.
- `embeddings/` owns configuration, raw-model HTTP transport, bounded slide
  processing, and incremental HDF5 output for GigaPath and Virchow2.
- `serve.py` hosts the three new standalone workflows through one configurable
  Ray ingress. `settings.py` defines deployment-only worker settings.
- `artifacts.py` publishes new workflow outputs atomically and prevents replacing
  their input files.
- `configuration.py` enforces complete resolved settings for all concrete schemas.
  `heatmap/configuration.py` shares spatial-map stride and TIFF settings without
  coupling EPISeg configuration to the prostate workflow.
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

See [the additional-workflow guide](../docs/guides/additional-workflows.md) for
inputs, output schemas, examples, sampling assumptions, and deployment requirements.

Workflow processing receives provider-neutral readers and returns outputs.
Provider configuration fetches, publication, and finalization remain in
`../integrations/`; scheduling and execution state remain in
`../execution_scheduler/`.
