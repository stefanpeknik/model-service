# Model Service

RationAI's service for running pathology models on Ray Serve: raw per-tile
inference, and **builders** that run a chosen model over a whole slide, either
synchronously or as durable executions you can poll and cancel.

```text
deploy/helm/rayservice/models/<id>.yaml   one file per trained model: inference contract + Ray Serve app
src/model_service/models/                            inference implementations (classifier, segmentation, encoders)
src/model_service/builders/                          heatmap and tile-embeddings builders and their runtime
src/model_service/facade/                     the public API (facade) and model catalog
src/model_service/scheduler/               durable executions on PostgreSQL
src/model_service/providers/                      providers: shared storage, EMPAIA
```

Read [how it is put together](docs/architecture/core.md), then the
[documentation map](docs/README.md).

## Add a model

A model that fits an existing implementation needs only YAML: add
`deploy/helm/rayservice/models/<id>.yaml`, list it under `models:` in the values file,
and `helm upgrade`. See [Adding models](docs/guides/adding-models.md).

## Use it

```python
from rationai import Client
from rationai.types import BuilderRequest, InputAccess, SharedPathInput

with Client(base_url="http://<facade>:8080") as client:
    print([(m.id, m.builders) for m in client.models.list()])
    scheduled = client.builders.schedule(
        "heatmap",
        BuilderRequest(
            model_id="prostate-classifier-1",
            inputs={"slide": SharedPathInput("slides/slide.mrxs")},
            input_access=InputAccess("shared_storage", {"root": "/mnt/data"}),
            output_path="outputs/heatmap.tiff",
        ),
    )
    print(client.executions.get(scheduled.execution.id).status)
```

## Develop

```sh
UV_CACHE_DIR=.cache/uv uv sync
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
