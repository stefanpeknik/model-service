# Model Service

RationAI's service for running pathology models on Ray Serve: **builders** run
a chosen model over a whole slide, either
synchronously or as durable executions you can poll, cancel, or leave queued
until a GPU frees up.

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
  api/                    the public API process: routes/, synchronous runs
  http_common/            auth, error mapping, scheduling calls (API and provider routes)
  scheduler/              durable executions on PostgreSQL (runs inside the API process)
  builders/               builder contract, registry, shared slides/
  builders/implementations/  the builders: heatmap/, embeddings/
  builders/replica/       the Ray Serve app that hosts builders
  models/                 model catalog, one client per contract, MLflow loader
  providers/              where builders read input and publish output: standalone, empaia/
  contracts/              data shapes processes hand each other
  models/implementations/    inference code served by Ray Serve
deploy/
  helm/rayservice/models/<id>.yaml   one file per trained model
  images/                            Dockerfiles
  k8s/                               manifests the chart does not create
testing/         deployable test doubles (constant classifier, mock AppService)
tests/           unit and PostgreSQL tests
```

Start with [what this thesis added](docs/thesis-scope.md), then
[Adding models](docs/guides/adding-models.md) or the
[EMPAIA integration](docs/architecture/empaia.md).

## Add a model

A model that fits an existing implementation needs only YAML: add
`deploy/helm/rayservice/models/<id>.yaml`, list it under `models:` in the values file,
and `helm upgrade`. See [Adding models](docs/guides/adding-models.md).

## Add a builder

A new slide-level output is a builder package under
`src/model_service/builders/` (a declaration and a `Builder`), one line in
`registry.py` and an entry under `builders:` in the values file. A new kind of
model adds an inference contract and its model client. See
[Adding a builder or an inference contract](docs/guides/adding-builders.md).

## Use it

```python
from rationai import Client
from rationai.types import BuilderRequest, InputAccess, SharedPathInput

with Client(base_url="http://<model-service>:8080") as client:
    print([(m.id, m.builders) for m in client.models.list()])
    scheduled = client.builders.schedule(
        "heatmap",
        BuilderRequest(
            model_id="prostate-classifier-1",
            inputs={"slide": SharedPathInput("slides/slide.mrxs")},
            input_access=InputAccess("shared_storage", {"root": "/mnt/data"}),
            output_path="outputs/heatmap.npz",
        ),
    )
    print(client.executions.get(scheduled.execution.id).status)
```

For one raw tile, skip the API: post a single LZ4-compressed
`tile_size`×`tile_size`×3 uint8 tile to the model's Ray Serve ingress,
`POST http://<ray-serve>:8000/<model-id>/`. `GET /v1/models/<id>` gives the
model's `tile_size` and contract.

The heatmap is a score grid: `numpy.load("heatmap.npz")` gives the scores per
cell (`values`), which cells were scored, and the slide level, cell size and
microns per pixel to place them on the slide.

## Develop

```sh
UV_CACHE_DIR=.cache/uv uv sync --all-extras
podman run -d --name ms-test-pg -p 127.0.0.1:55432:5432 \
  -e POSTGRES_PASSWORD=test docker.io/library/postgres:17-alpine
TEST_POSTGRES_DSN=postgresql://postgres:test@127.0.0.1:55432/postgres \
  UV_CACHE_DIR=.cache/uv uv run python -m unittest discover -s tests -t .
UV_CACHE_DIR=.cache/uv uv run ruff check
UV_CACHE_DIR=.cache/uv uv run mypy src testing
```

Without `TEST_POSTGRES_DSN` the PostgreSQL tests are skipped.

## License

MIT; see `LICENSE`. Developed by the RationAI team at Masaryk University,
Faculty of Informatics.
