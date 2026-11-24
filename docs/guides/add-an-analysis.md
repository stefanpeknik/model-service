# Add an analysis

An analysis is one slide-in, results-out job, written as one YAML file in
`catalog/analyses/`. A source produces items from the slide, steps transform
them (a `{model: <id>}` step calls a served model on each tile), and sinks
collect the results into files of a [data format](../reference/formats.md).
The file name is the analysis's ID: `POST /v1/analyses/<id>/run` runs it.

You need a served model for it to call. If yours isn't deployed yet,
[add it](add-a-model.md) first.

## 1. Write the file

`catalog/analyses/prostate-heatmap.yaml` without its EMPAIA section is the
whole heatmap job:

```yaml
# yaml-language-server: $schema=../.schemas/analyses.schema.json
summary: Scores each tile of a prostate slide into a heatmap.
source: {use: slide_tiles, stride_fraction: 0.5, skip_background: true, min_tissue_fraction: 0.0}
steps:
  - {model: prostate-classifier-1}
sinks:
  - {use: score_grid, output: heatmap}
```

Copy it to `catalog/analyses/my-heatmap.yaml`, keep the first line, and
change what differs:

- `steps` names your served model. The source reads tiles at the first model
  step's `tile_size` and `mpp`, so the geometry can't drift from what the
  model expects.
- `source` options set the defaults of the run's settings.
  `stride_fraction: 1.0` places tiles side by side, 0.5 overlaps them by half.
  `region_of_interest: required` makes callers send a mask.
- `sinks` names what the run writes. `score_grid` averages each tile's
  scores into a grid over the slide; `embedding_table` writes one feature row
  per tile and needs an encoder, a model with `output: {kind: embedding}`.

[Components](../reference/components.md) lists every source, step and sink
with its options and the columns it reads and adds.

## 2. Check it

```sh
uv run pytest tests/catalog
```

The check follows the columns from the source through each step to the
sinks, with the real shapes of your models, and reports anything that
doesn't fit. An embedding sink after a classifier, for example:

```text
catalog/analyses/my-heatmap.yaml:sinks[0]: `embedding_table` reads `features`, which nothing before it adds. Added by: a model step with `output: {kind: embedding, …}`. Hint: put a component that adds `features` before it
```

A misspelt model ID gets the list of served models that exist. Once the
check passes, regenerate the reference docs and commit them with the file:

```sh
PYTHONPATH=src uv run tools/generate.py
```

## 3. Test it

`uv run pytest tests/catalog` is the quick check while you edit. Before you
deploy, run the whole suite once:

```sh
uv run pytest
```

It needs the development environment from the README's
[Develop](../../README.md#develop) section: `uv sync` with the process
extras, and a PostgreSQL container if you want the scheduler tests to run
rather than skip. With `helm` installed, the suite renders the chart with
your analysis, and it checks the generated docs, so it catches mistakes the
catalog check can't see. Nothing else checks the file before the cluster
loads it.

## 4. Deploy and run it

Add the ID to `analyses` in the environment's values file if it names IDs,
as `environments/rationai-test/values.yaml` does, and `helm upgrade`. The chart refuses to
render an analysis whose served model the release doesn't deploy.

From Python, with the RationAI SDK:

```python
from rationai import Client, formats
from rationai.types import AnalysisRequest, InputAccess, SharedPathInput

with Client(base_url="http://<model-service>:8080") as client:
    result = client.analyses.run(
        "my-heatmap",
        AnalysisRequest(
            inputs={"slide": SharedPathInput("slides/slide.mrxs")},
            input_access=InputAccess("shared_storage", {"root": "/mnt/data"}),
            output_path="outputs/my-heatmap.npz",
            configuration={"stride_fraction": 1.0},
        ),
    )

grid = formats.read(result.outputs["heatmap"].path, result.outputs["heatmap"].media_type)
```

`client.analyses.schedule` takes the same request and runs it as a durable
execution you can poll and cancel. `configuration` overrides settings for one
run. The settings are the source's and steps' per-run options, and
[Analyses](../reference/analyses.md) lists each analysis's settings with
their defaults.

## 5. Offer it in EMPAIA

An `expose.empaia` section turns the analysis into an EMPAIA app, which
model-service registers when the control plane starts:

```yaml
expose:
  empaia:
    name: Prostate cancer heatmap
    name_short: ProstHM
    description: Classifies prostate tissue and shows the scores as a heatmap over the slide.
    modes:
      standalone: {parameters: [stride_fraction]}
      preprocessing:
        triggers:
          - {tissue: PROSTATE, stain: H_AND_E}
    outputs:
      heatmap: {type: continuous_pixelmap}
```

Each output's `type` picks the publisher that carries it to EMPAIA.
`parameters` are the settings a pathologist may change in a standalone job.
The loader checks what EMPAIA would refuse later. A continuous pixelmap needs
one score channel, preprocessing can't start an analysis that requires a
region of interest, and the app name has at most 30 characters.
`prostate-heatmap.yaml` shows the viewer settings too, and
[EMPAIA integration](../architecture/empaia.md) explains how jobs flow.
