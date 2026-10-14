# Model-service core: builders, model catalog and durable execution

Status: active, 2026-10-14. Branch `claude/model-service-core`.

This is the specification for finishing model-service so that EMPAIA (or any
other caller) can treat it as a black-box service. It keeps the decisions the
user approved in the generic-computation map (tickets 01–04) and settles the
open ones (05–08) with the smallest mechanism that meets them. Tickets live in
`~/.claude/.scratch/model-service/core-completion/issues/`.

## Problem

- Adding a model that fits an existing builder needs Python edits. The
  per-model workflow wrappers (`prostate-heatmap`, `episeg-segmentation`,
  `gigapath-embeddings`, `virchow2-embeddings`) hardcode model names, even though
  `models/` and the original heatmap builder were already generic.
- The scheduler processes one execution at a time per process, and limits
  capacity per workflow rather than per model.
- Several correctness and safety gaps exist: caller-chosen storage roots,
  caller-chosen EMPAIA URLs, EMPAIA jobs left non-terminal after a failed
  publication, unbounded EMPAIA workspaces, embeddings read at level 0 regardless
  of the model's resolution, unpinned model code loaded from GitHub `main`.

## Terms

- **Model deployment**: one trained model served by Ray, identified by its
  model ID (the Ray application name, e.g. `prostate-classifier-1`).
- **Inference contract**: the model-ready input and prediction structure a
  model deployment declares. Two exist: `spatial-scores` and `tile-embedding`.
- **Builder**: reusable whole-slide processing that calls one compatible model
  deployment and writes declared outputs. Two exist: `heatmap` and
  `tile-embeddings`.
- **Execution**: the durable lifecycle of one scheduled builder invocation.

"Workflow" is retired from code, routes and the SDK.

## Decisions

### Model catalog from deployment YAML

Each model deployment is one file, `helm/rayservice/models/<model-id>.yaml`:

```yaml
inference:
  contract: tile-embedding
  mpp: 0.5
  summaries: [global, local_mean]
serve:            # a Ray Serve application, injected into serveConfigV2 unchanged
  name: virchow2
  route_prefix: /virchow2
  import_path: models.virchow2:app
  deployments: [...]
```

Helm renders every `serve` block into the RayService and the whole list into a
`model-catalog` ConfigMap mounted into the facade. A checksum annotation
restarts the facade when the catalog changes; the scheduler is durable, so a
restart is safe. The facade validates the catalog at startup with Pydantic and
refuses to start on an invalid entry.

Contract parameters come from the `serve` block's first deployment
`user_config` where the model already reads them, and from `inference` for
values the model does not read. Nothing is declared twice.

| Contract | Transport | Parameters | Source |
| --- | --- | --- | --- |
| `spatial-scores` | Ray handle: `get_config()`, `predict(CHW uint8)` → `n_channels × output_tile_size²` scores in [0, 1] | `tile_size`, `output_tile_size`, `n_channels`, `mpp` | `user_config` |
| `tile-embedding` | HTTP `POST /{id}/`: LZ4 HWC uint8 tile, `x-output-dtype: float32`, `x-pool-tokens: true` → float32 `(dim,)` or `(len(summaries), dim)` | `tile_size` from `user_config`; `mpp`, `summaries` from `inference` | both |

Onboarding a compatible model: add one YAML file, list it in `values.yaml`
`models`, `helm upgrade`. No Python, no image rebuild.

### Builders

Builders are Python, registered in `builders/registry.py`. Each declares the
contract it accepts, its inputs and outputs, and a Pydantic configuration with
defaults. One Ray application per builder (`builders.serve:app`, route
`/builders/<id>`), with the builder ID in `user_config`.

| Builder | Accepts | Inputs | Output | Defaults |
| --- | --- | --- | --- | --- |
| `heatmap` | `spatial-scores` | `slide`, optional `region_of_interest` mask | `heatmap` pyramidal TIFF | stride 0.5, 256² TIFF tiles, `uint8` encoding |
| `tile-embeddings` | `tile-embedding` | `slide`, optional `region_of_interest` mask | `embeddings` HDF5 | stride 1.0 |

- The facade resolves the model's current catalog entry for every attempt and
  sends it with the invocation. Builders never read the catalog.
- `heatmap` checks the model's `get_config()` against the declared parameters
  and fails with the non-retryable `model_contract_mismatch` on disagreement.
- `heatmap` keeps `uint8` as its default because it preserves the existing
  prostate artifacts and the EMPAIA publisher's input. `float32` is a builder
  setting.
- `tile-embeddings` reads tiles at the slide level closest to the model's
  `mpp`, and records level, downsample and both resolutions in the HDF5.
- Without a mask, every grid tile is processed.

### Public HTTP

| Route | Purpose |
| --- | --- |
| `GET /v1/models` | Catalog: ID, contract, parameters, compatible builders |
| `GET /v1/builders` | Builders: ID, accepted contract, inputs, outputs |
| `POST /v1/builders/{builder}/run` | Synchronous run. Body: `model_id`, `inputs`, `input_access`, optional `output_path`, `deadline_at` |
| `POST /v1/builders/{builder}/schedule` | Durable admission. Body: `provider`, `body` (with `model_id`), `idempotency_key`, `requested_timeout_seconds` |
| `POST /v1/providers/empaia/builders/{builder}/schedule` | EMPAIA admission with `model_id` and job context |
| `GET /v1/executions/{id}`, `POST …/cancel` | Unchanged; views carry `builder_id` and `model_id` |
| `POST /{model_id}/…` | Raw inference proxy, unchanged |

Unknown builder or model: 404 `unknown_builder` / `unknown_model`.
Incompatible pair: 422 `incompatible_model`. Model temporarily down: durable
requests queue; synchronous requests get 503.

### Durable execution

- One process runs up to `MAX_CONCURRENT_EXECUTIONS` claims (default 4).
- The capacity limit is per model (`PER_MODEL_LIMIT`, default 1), because the
  GPU-bound model is the shared resource.
- Lease renewal tolerates transient database errors until the lease would
  actually expire.
- Migration `0002` renames `workflow_id` to `builder_id` and adds `model_id`,
  mapping existing rows from the retired workflow IDs.
- Scheduled raw inference is deferred. Nothing in either assignment requires
  it, and it would need durable payload storage. Raw calls stay transient.

### Safety and correctness fixes

- Shared-storage `root` must be one of the configured `SHARED_STORAGE_ROOTS`.
- EMPAIA `api_url` must match a configured `EMPAIA_API_URLS` prefix.
- A failed EMPAIA publication reports failure to EMPAIA instead of leaving the
  job open; idempotent pixelmap tile writes are retried.
- EMPAIA attempt workspaces are removed when the attempt ends.
- All Ray applications run the code baked into the image; no `working_dir`
  download from GitHub.

### Removed

Per-model workflow modules and serve entry points, `workflow-definitions/`,
the unsafe legacy `builders/heatmap_builder.py` endpoint, and
`prototypes/empaia-control-plane/`.

## Acceptance

1. Unit and PostgreSQL-backed tests pass locally.
2. Live in `peknik-ns`, one GPU worker at a time, scaled to zero afterwards:
   - prostate heatmap via `heatmap` + `prostate-classifier-1`, synchronous and
     scheduled, pixel-identical to the October 8 artifact;
   - a second model onboarded by YAML only (no Python change, no image
     rebuild) and run through the same builder;
   - `tile-embeddings` with one embedding model on the same slide;
   - cancellation and scheduler restart during a running execution.
3. The RationAI SDK runs and schedules builders with a model ID.

## Out of scope

EMPAIA job runs, JES integration, app registration, annotation outputs and
PREPROCESSING mode. Those start once this core is accepted. EMPAIA code keeps
working against its mock AppService tests.
