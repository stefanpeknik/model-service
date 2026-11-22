# Model Service

RationAI's service for running pathology models on Ray Serve: **analyses** run
served models over a whole slide, either synchronously or as durable
executions you can poll, cancel, or leave queued until a GPU frees up.

## In short

- **What it does:** turns a trained model (an ONNX checkpoint in MLflow, or a foundation model) into a
  service that scores whole slides, for RationAI's own tools and for
  researchers working in EMPAIA.
- **Adding a model:** a model that fits an existing implementation is one YAML
  file and a `helm upgrade`; no Python. See [Add a model](#add-a-model).
- **EMPAIA:** model-service registers itself as an EMPAIA app and runs EMPAIA
  jobs. On the EMPAIA test instance a prostate heatmap job finishes in 49 s
  warm, 101 s with a cold GPU, and the result shows in the case browser
  ([run record](docs/operations/empaia-score-grid-2026-10-31.md)).
- **Status (2026-11):** running on the RationAI test namespace beside the
  EMPAIA test instance; not yet in production.
- **What this thesis added** to the original model-service:
  [thesis scope](docs/thesis-scope.md).

## Status and evidence

Dated records of live runs, newest first (all in
[docs/operations/](docs/operations/)):

- [2026-11-01](docs/operations/empaia-provider-composition-2026-11-01.md):
  self-registration in EMPAIA, one app with both modes, heatmap in the case browser.
- [2026-10-31](docs/operations/empaia-score-grid-2026-10-31.md): prostate heatmap
  published to EMPAIA with timings.
- [2026-10-25](docs/operations/empaia-test-instance-2026-10-25.md): first job
  started from the EMPAIA test instance.
- [2026-10-14](docs/operations/core-acceptance-2026-10-14.md): core API and
  scheduler acceptance.

Both EMPAIA modes run live. A standalone (region-of-interest) result first did
not show in the case browser; its maintainers fixed that on 2026-11-11, and a
later standalone run showed its result.

## Layout

```text
src/model_service/
  api/                    the public API process: routes/v1/, mounted catalog, synchronous runs
  api/http/               auth, error mapping, scheduling calls (API and provider routes)
  api/scheduler/          durable executions on PostgreSQL (runs inside the API process)
  served_model/           the generic host that serves every catalog/served-models/ file
  analysis_runner/        the one Ray Serve app that runs every catalog/analyses/ file
  catalog/, components/   catalog loading and the components catalog files name
  formats/                data formats sinks write and readers open
  slides/                 slide reading and tissue detection for analysis sources
  providers/              where runs read input and publish output: standalone, empaia/
  contracts/              data shapes processes hand each other
deploy/
  helm/rayservice/                   the chart; reads catalog/ through a symlink
  images/                            Dockerfiles
  k8s/                               manifests the chart does not create
catalog/
  served-models/<id>.yaml            one file per served model
  analyses/, environments/<env>/     analyses; per-environment patches
tests/           unit and PostgreSQL tests
```

Start with [what this thesis added](docs/thesis-scope.md), then
[Adding models](docs/guides/adding-models.md) or the
[EMPAIA integration](docs/architecture/empaia.md).

## Add a model

A model the existing components can run needs only YAML: add
`catalog/served-models/<id>.yaml`, check it with `uv run pytest tests/catalog`,
and `helm upgrade`; every file is deployed unless the environment's values
name a subset in `servedModels`. See the
[Configuration Reference](docs/guides/configuration-reference.md) for the
file's blocks.

## Add an analysis

A new slide-level output is a file in `catalog/analyses/`: a source, steps
(`{model: <id>}` calls a served model) and sinks, by component name. Check it
with `uv run pytest tests/catalog` and `helm upgrade`. Code is needed only for
a missing component, under `src/model_service/components/`.

## Use it

```python
from rationai import Client, formats
from rationai.types import AnalysisRequest, InputAccess, SharedPathInput

with Client(base_url="http://<model-service>:8080") as client:
    print([(a.id, a.models) for a in client.analyses.list()])
    print([(m.id, m.analyses) for m in client.models.list()])
    scheduled = client.analyses.schedule(
        "prostate-heatmap",
        AnalysisRequest(
            inputs={"slide": SharedPathInput("slides/slide.mrxs")},
            input_access=InputAccess("shared_storage", {"root": "/mnt/data"}),
            output_path="outputs/heatmap.npz",
        ),
        idempotency_key="heatmap-slide-001",
    )
    (execution,) = client.executions.wait([scheduled.execution.id])
    print(execution.status, execution.analysis_digest)

heatmap = execution.result.outputs["heatmap"]
grid = formats.read(heatmap.path, heatmap.media_type)
```

`client.analyses.run` takes the same request and waits for the outputs.
`formats.read` refuses a format version it doesn't know; reading an
`embedding_table` needs the SDK's `hdf5` extra.

For one raw tile, skip the API: post a single LZ4-compressed
`tile_size`×`tile_size`×3 uint8 tile to the model's Ray Serve ingress,
`POST http://<ray-serve>:8000/<model-id>/`. `GET /v1/models/<id>` gives the
model's declaration, with `tile_size` and `mpp` under `input`.

The heatmap is a score grid (`score_grid` v1): its scores per cell (`values`),
which cells were scored, and the slide level, cell size and microns per pixel
to place them on the slide. Without the SDK, `numpy.load("heatmap.npz")`
reads it too.

## Develop

```sh
# The SDK's models submodule is read from GitLab; see [tool.uv.sources].
export GIT_CONFIG_COUNT=1 \
  GIT_CONFIG_KEY_0=url.git@gitlab.ics.muni.cz:rationai/digital-pathology/production/services/models.git.insteadOf \
  GIT_CONFIG_VALUE_0=https://github.com/AigoPath/EMPAIA-Models.git
UV_CACHE_DIR=.cache/uv uv sync --extra api --extra builders --extra inference --extra cpu
podman run -d --name ms-test-pg -p 127.0.0.1:55432:5432 \
  -e POSTGRES_PASSWORD=test docker.io/library/postgres:17-alpine
TEST_POSTGRES_DSN=postgresql://postgres:test@127.0.0.1:55432/postgres \
  UV_CACHE_DIR=.cache/uv uv run pytest
UV_CACHE_DIR=.cache/uv uv run ruff check
UV_CACHE_DIR=.cache/uv uv run mypy .
```

Without `TEST_POSTGRES_DSN` the PostgreSQL tests are skipped. Tests marked
`live` call the deployed cluster and run only with `uv run pytest -m live`.
`python -m unittest discover -s tests -t .` still runs the unittest-style tests.

## License

MIT; see `LICENSE`. Developed by the RationAI team at Masaryk University,
Faculty of Informatics.
