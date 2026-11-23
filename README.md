# Model Service

RationAI's service for running pathology models on Ray Serve. Trained models
are **served models**; whole-slide jobs that call them are **analyses**. Both
are YAML files in `catalog/`, built from reusable **components**. An analysis
runs synchronously, as a durable execution you can poll and cancel, or as an
EMPAIA app.

## In short

- **A new model** that an existing runtime can run (an ONNX graph, a timm or
  Hugging Face encoder) is one file in `catalog/served-models/` and a
  `helm upgrade`, with no Python and no image rebuild.
  [Add a model](docs/guides/add-a-model.md).
- **A new analysis** is one file in `catalog/analyses/`: a source, steps
  (`{model: <id>}` calls a served model) and sinks.
  [Add an analysis](docs/guides/add-an-analysis.md).
- **Code** is needed only for a missing component, one small class with its
  own test. [Write a component](docs/guides/write-a-component.md).
- **Every catalog file is checked** before it deploys:
  `uv run pytest tests/catalog` lists every mistake with the file, the YAML
  path and a hint. Editors with the YAML language server complete keys and
  flag errors from the generated JSON Schemas in `catalog/.schemas/`.
- **EMPAIA:** an analysis's `expose.empaia` section makes it an EMPAIA app,
  which model-service registers itself. The prostate heatmap app runs on the
  EMPAIA test instance; a job takes 49 s warm and 101 s with a cold GPU
  ([run record](docs/operations/empaia-score-grid-2026-10-31.md)).
- **Status (2026-11):** running on the RationAI test namespace beside the
  EMPAIA test instance; not yet in production. What the thesis added to the
  original model-service: [thesis scope](docs/thesis-scope.md).

## Start here

You'll write in two places. YAML goes in `catalog/`, one file per served
model or analysis. Code goes in `src/components/`, one small class per file,
in the folder that says where it may be used. Copy a template from
`src/components/_templates/` and subclass a kind from `src/components/base/`.
Everything in `src/model_service/` is the plumbing that loads, checks and runs
those two, and a new model or analysis shouldn't need to touch it.

## Reference

Generated from the code and the catalog by `tools/generate.py`, so they
match what deploys:

- [Components](docs/reference/components.md): every loader, runtime, step,
  source, sink, converter and publisher, with its options and where it may be
  used;
- [Served models](docs/reference/served-models.md) and
  [analyses](docs/reference/analyses.md) in `catalog/`;
- [Data formats](docs/reference/formats.md) of the files analyses write.

[How model-service is put together](docs/architecture/core.md) explains the
design, and the [documentation map](docs/README.md) lists the rest.

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
`embedding_table` needs the SDK's `hdf5` extra. Without the SDK,
`numpy.load("heatmap.npz")` reads a score grid too.

For one tile, skip the API and post it to the served model's Ray Serve route;
[raw inference](docs/available-models.md) shows the request and reply.

## Layout

```text
catalog/
  served-models/<id>.yaml            one file per served model
  analyses/<id>.yaml                 one file per analysis
  environments/<env>/                per-cluster patches of those files
  .schemas/                          generated JSON Schemas
src/components/           what you write
  steps/                  pre/post steps of a served model and steps of an analysis
  loaders/  runtimes/     where weights come from and what runs them (served models)
  sources/  sinks/        where an analysis's items come from and what it writes
  empaia/                 converters/ and publishers/ for EMPAIA inputs and outputs
  base/                   the kind classes you subclass, Options, Col, Run, Problem
  _templates/             one file per kind to copy
  testing/                fake slides and models, run_component, conformance suites
src/model_service/        the plumbing
  catalog/                loading and checking catalog files
  registry/               the scan that finds components and checks their folders
  served_model/           the generic host that serves every served model
  analysis_runner/        the Ray Serve app that runs every analysis
  formats/                data formats sinks write and readers open
  api/                    the public API process, with the durable scheduler in api/scheduler/
  providers/              where runs read input and publish output: standalone, empaia/
  slides/                 slide reading and tissue detection for sources
  contracts/              data shapes processes hand each other
  reference/              what tools/generate.py writes
deploy/helm/rayservice/   the chart; reads catalog/ through a symlink
docs/                     guides, generated reference, architecture, dated operations records
tests/                    unit, catalog, chart and PostgreSQL tests; tests/live/ for the cluster
tools/                    generate.py for the reference; warm_model.py, request_preprocessing.py for the cluster
```

## Develop

```sh
# The SDK's models submodule is read from GitLab; see [tool.uv.sources].
export GIT_CONFIG_COUNT=1 \
  GIT_CONFIG_KEY_0=url.git@gitlab.ics.muni.cz:rationai/digital-pathology/production/services/models.git.insteadOf \
  GIT_CONFIG_VALUE_0=https://github.com/AigoPath/EMPAIA-Models.git
export UV_CACHE_DIR=.cache/uv
uv sync --extra api --extra analysis-runner --extra served-models --extra cpu
podman run -d --name ms-test-pg -p 127.0.0.1:55432:5432 \
  -e POSTGRES_PASSWORD=test docker.io/library/postgres:17-alpine
TEST_POSTGRES_DSN=postgresql://postgres:test@127.0.0.1:55432/postgres uv run pytest
uv run ruff check
uv run mypy src
PYTHONPATH=src uv run tools/generate.py   # after changing components, formats or catalog/
```

Each process has its extra: `api`, `analysis-runner` and `served-models`,
with `cpu` or `gpu` for the runtimes. Without `TEST_POSTGRES_DSN`
the PostgreSQL tests are skipped. Tests marked `live` call the deployed
cluster and run only with `uv run pytest -m live`.

## License

MIT; see `LICENSE`. Developed by the RationAI team at Masaryk University,
Faculty of Informatics.
