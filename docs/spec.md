# Model-service core: builders, model catalog and durable execution

Status: implemented and accepted live on 2026-10-14
([evidence](operations/core-acceptance-2026-10-14.md)); scheduled raw inference
added and accepted on 2026-10-15
([evidence](operations/raw-inference-acceptance-2026-10-15.md)). Branch
`claude/model-service-core`.

This is the specification for finishing model-service so that EMPAIA (or any
other caller) can treat it as a black-box service. It keeps the decisions the
user approved in the generic-computation map (tickets 01–04) and settles the
open ones (05–08) with the smallest mechanism that meets them. Tickets live in
`~/.claude/.scratch/model-service/core-completion/issues/`.

## Problem

- Adding a model that fits an existing builder needs Python edits. The
  per-model workflow wrappers (`prostate-heatmap`, `episeg-segmentation`,
  `gigapath-embeddings`, `virchow2-embeddings`) hardcode model names, even though
  `src/model_service/models/` and the original heatmap builder were already generic.
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

Each model deployment is one file, `deploy/helm/rayservice/models/<model-id>.yaml`:

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
| `tile-embedding` | Ray handle: `get_config()` (`tile_size`, `summaries`), `embed(CHW uint8)` → float32 `(dim,)` or `(len(summaries), dim)` | `tile_size` from `user_config`; `mpp`, `summaries` from `inference` | both |

Builders call every model by deployment handle, never over HTTP, and check
`get_config()` against the catalog before reading a slide. A test checks that
each model file's implementation has the handle methods of its contract. The
raw HTTP route (`POST /{id}/`) stays for external callers and wraps the same
code, so both give identical outputs.

Onboarding a compatible model: add one YAML file, list it in `values.yaml`
`models`, `helm upgrade`. No Python, no image rebuild.

### Builders

Builders are Python, registered in `src/model_service/builders/registry.py`. Each declares the
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
| `POST /v1/builders/{builder}/models/{model}/run` | Synchronous run. Body: `inputs`, `input_access`, optional `output_path`, `deadline_at` |
| `POST /v1/builders/{builder}/models/{model}/schedule` | Durable admission with shared-storage inputs. Body: `inputs`, `input_access`, optional `output_path`, `idempotency_key`, optional `requested_timeout_seconds` |
| `POST /v1/empaia/builders/{builder}/models/{model}`, `GET`/`DELETE /v1/empaia/executions/{id}` | EMPAIA jobs from JES's HTTP execution backend (app registered as `http::<target>/builders/<builder>/models/<model>`): the body is what an app container gets (App API URL, job ID, token); the job's mode, inputs and configuration come from the App API; the `Location` is polled and cancelled. A failure or cancellation reads as `running` until EMPAIA has been told; a running job's `message` explains a wait for capacity |
| `POST /v1/models/{model}/run` | Raw inference: one LZ4-compressed RGB tile of the model's `tile_size`, optional `x-output-dtype`/`x-pool-tokens`; checked, then the model's response is returned. Replaces the root `POST /{model}/` proxy |
| `POST /v1/models/{model}/schedule` | Durable raw inference: the body `run` takes, an `Idempotency-Key` header, optional `timeout_seconds` |
| `GET /v1/executions/{id}`, `POST …/cancel` | Views carry `operation` (`builder` or `inference`), `builder_id` (null for inference), `model_id`, `provider_pending` (a failure or cancellation not yet reported to the provider) and `message` (see below) |
| `POST /v1/executions/lookup` | Up to 1000 views by ID in one call; unknown IDs are left out |
| `GET /v1/executions/{id}/result` | A succeeded inference execution's stored model response |

Unknown builder or model: 404 `unknown_builder` / `unknown_model`.
Incompatible pair: 422 `incompatible_model`. Model temporarily down: durable
requests queue; synchronous requests get 503.

EMPAIA admission asks AppService for each optional builder input and leaves out
those the job does not have (AppService answers 404). A preprocessing app, which
EMPAIA starts with only a slide, thus runs the heatmap builder over the whole
slide. An app file may list the inputs it takes (`inputs: [slide]`, see
`deploy/empaia/apps/constant-heatmap-preprocessing.yaml`).

EMPAIA slide inputs reach builders one of two ways, chosen by
`EMPAIA_SLIDE_ACCESS` on the Ray workers (Helm
`executionScheduler.empaia.slideAccess`):

- `direct` (default): link the file named by the RationAI direct-access route.
- `http`: download the slide through the App API's region route into the
  attempt workspace, for installations whose slide storage model-service cannot
  mount (the EMPAIA test instance's slides sit on a volume in another
  namespace). Only the level closest to the model's `mpp` is fetched, tile by
  tile and streamed into a tiled JPEG TIFF that records that level's
  resolution, so the builder reads it unchanged as level 0. Region-of-interest
  rectangles still become a local mask. The model's `mpp` reaches the input
  adapter with the invocation.

A running execution whose model or builder has no running Serve replica, only
one Ray has not placed on a node for `CAPACITY_GRACE_SECONDS` (default 60),
keeps `running` with `message: "waiting for capacity: no replica of model <id>
could be scheduled for <N>s"`. Ray Serve keeps such a replica indefinitely and
calls to it wait, so the message is the only sign; the execution's deadline
still applies. The scheduler reads the Ray dashboard's
`/api/serve/applications/` (`RAY_DASHBOARD_ORIGIN`, set by Helm), cached for
10 s; without it, or when it is unreachable, there is no message. Background:
Ray Serve's default autoscaling policy counts only running replicas, so it
never scales down a replica that was never placed.

### Durable execution

- One process runs up to `MAX_CONCURRENT_EXECUTIONS` claims (default 4).
- The capacity limit is per model (`PER_MODEL_LIMIT`, default 1), because the
  GPU-bound model is the shared resource.
- Lease renewal tolerates transient database errors until the lease would
  actually expire.
- Database connections are pooled (`DATABASE_POOL_SIZE`, 20). A new
  connection per transaction made PostgreSQL the bottleneck under many small
  executions.
- Migration `0002` renames `workflow_id` to `builder_id` and adds `model_id`,
  mapping existing rows from the retired workflow IDs.
- Raw inference can be scheduled too; see below.

### Scheduled raw inference

A caller that tiles slides itself should not hold one connection per tile
while a GPU worker starts. `POST /v1/models/{model}/schedule` takes exactly the
request the raw route takes and returns an execution.

- One execution per tile, the native inference unit. It shares the builder
  lifecycle (leases, fencing, retries, deadlines, cancellation, idempotent
  admission) but has no provider phase: there is nothing to publish or report.
- Every model's raw input is one LZ4 frame holding an HWC uint8 RGB tile of the
  model's `tile_size`. The facade checks that at admission and before every
  attempt, decompressing at most one byte past the expected size. A model whose
  tile size changed while a tile waited fails it with `incompatible_input`
  instead of sending reinterpreted data.
- The request bytes are stored in `inference_payloads` in the admission
  transaction, so an acknowledged tile survives a restart. They are deleted
  when the execution finishes. The response is kept for
  `INFERENCE_RETENTION_SECONDS` (24 h) after finishing, then the execution is
  deleted. `x-` response headers (e.g. `x-output-shape`) are kept with it.
- Replay identity is the `Idempotency-Key` header; the fingerprint covers the
  model, `x-` request headers, timeout and payload hash.
- Model responses: 2xx succeeds; 429/503 requeue without spending an attempt;
  400/413/415/422 fail with `model_rejected_input`; other statuses and lost
  connections are retryable failures.
- Tiles have their own capacity: `PER_MODEL_INFERENCE_LIMIT` (32) attempts per
  model and `MAX_CONCURRENT_INFERENCE` (32) per facade process, separate from
  builder slots, so neither starves the other. Ray Serve's own queue limits
  still protect the replica. The worker claims tiles in batches of a quarter of
  its tile slots per transaction.
- Facade memory for running tiles is bounded by
  `MAX_CONCURRENT_INFERENCE × (INFERENCE_MAX_REQUEST_BYTES + INFERENCE_MAX_RESULT_BYTES)`,
  192 MiB with the defaults (2 MiB and 4 MiB).
- Throughput is bounded by the facade's CPU, about 100 tiles/s per 1-CPU
  replica. Replicas scale it: leases and the per-model locks already
  coordinate them.

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
the unsafe legacy `src/model_service/builders/heatmap_builder.py` endpoint, and
`prototypes/empaia-control-plane/`.

## Acceptance

1. Unit and PostgreSQL-backed tests pass locally.
2. Live in `peknik-ns`, one GPU worker at a time, scaled to zero afterwards:
   - prostate heatmap via `heatmap` + `prostate-classifier-1`, synchronous and
     scheduled, pixel-identical to the previous implementation on the same
     inputs (the October 8 file turned out not to match those inputs);
   - a second model onboarded by YAML only (no Python change, no image
     rebuild) and run through the same builder;
   - `tile-embeddings` with one embedding model on the same slide;
   - cancellation and scheduler restart during a running execution.
3. The RationAI SDK runs and schedules builders with a model ID.
4. Scheduled raw inference (2026-10-15): tiles admitted while no GPU worker
   exists complete after it starts, without the client staying connected;
   results equal direct calls; a facade restart loses no tile.

## Out of scope

EMPAIA job runs, JES integration, app registration, annotation outputs and
PREPROCESSING mode. Those start once this core is accepted. EMPAIA code keeps
working against its mock AppService tests.
