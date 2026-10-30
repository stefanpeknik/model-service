# Model-service core: builders, model catalog and durable execution

Status: implemented and accepted live on 2026-10-14
([evidence](operations/core-acceptance-2026-10-14.md)); scheduled raw inference
added and accepted on 2026-10-15
([evidence](operations/raw-inference-acceptance-2026-10-15.md)); the EMPAIA
bridge ran end to end against the EMPAIA test instance on 2026-10-25
([evidence](operations/empaia-test-instance-2026-10-25.md)). Branch
`claude/empaia-http-protocol`.

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
- The heatmap builder skips tiles without tissue (`skip_background`, on by
  default): a tissue mask from the slide level closest to 8 µm/px, by colour
  saturation with an Otsu threshold, so white glass and transparent (black)
  regions are not sent to the model; skipped tiles score 0. It combines with a
  region-of-interest mask. Slides downloaded over the App API carry that coarse
  level as a second page. The tile-embeddings builder still processes every tile.

### Public HTTP

| Route | Purpose |
| --- | --- |
| `GET /v1/models` | Catalog: ID, contract, parameters, compatible builders |
| `GET /v1/builders` | Builders: ID, accepted contract, inputs, outputs |
| `POST /v1/builders/{builder}/models/{model}/run` | Synchronous run. Body: `inputs`, `input_access`, optional `output_path`, `deadline_at` |
| `POST /v1/builders/{builder}/models/{model}/schedule` | Durable admission with shared-storage inputs. Body: `inputs`, `input_access`, optional `output_path`, optional `requested_timeout_seconds`; an `Idempotency-Key` header |
| `POST /v1/empaia/builders/{builder}/models/{model}`, `GET`/`DELETE /v1/empaia/executions/{id}` | EMPAIA jobs from JES's HTTP execution backend (app registered as `http::<target>/builders/<builder>/models/<model>`): the body is what an app container gets (App API URL, job ID, token); the job's inputs and configuration come from the App API; submission is idempotent per job, and an `Idempotency-Key` header, when sent, must be the job ID (422 otherwise); the `Location` is polled and cancelled. A failure or cancellation reads as `running` until EMPAIA has been told; a running job's `message` explains a wait for capacity |
| `POST /v1/models/{model}/run` | Raw inference: one LZ4-compressed RGB tile of the model's `tile_size`, optional `x-output-dtype`/`x-pool-tokens`; checked, then the model's response is returned. Replaces the root `POST /{model}/` proxy |
| `POST /v1/models/{model}/schedule` | Durable raw inference: the body `run` takes, an `Idempotency-Key` header, optional `timeout_seconds` |
| `GET /v1/executions/{id}`, `POST …/cancel` | Views carry `operation` (`builder` or `inference`), `builder_id` (null for inference), `model_id`, `provider_pending` (a failure or cancellation not yet reported to the provider) and `message` (see below) |
| `POST /v1/executions/lookup` | Up to 1000 views by ID in one call; unknown IDs are left out |
| `GET /v1/executions/{id}/result` | A succeeded inference execution's stored model response |

Unknown builder or model: 404 `unknown_builder` / `unknown_model`.
Incompatible pair: 422 `incompatible_model`. Model temporarily down: durable
requests queue; synchronous requests get 503.

Authentication and authorization are interfaces with allow-everyone defaults
(`src/model_service/facade/auth.py`); no implementation is chosen yet. Every
route except the health checks asks an `Authenticator` for the caller's
`Principal` (401 `unauthenticated` when it refuses) and an `Authorizer` whether
that principal may perform the route's `Action`, given the execution ID in the
path (403 `forbidden`). The public API and the EMPAIA route each take their own
pair (`create_app(access=..., empaia_access=...)`): EMPAIA's service token, e.g.
checked like `empaia-receiver-auth` does, and direct users with owned executions
behind RBAC need different checks. Executions do not record an owner yet.

EMPAIA admission asks AppService for each optional builder input and leaves out
those the job does not have (AppService answers 404). That is all the job's mode
(standalone, preprocessing, …) changes for a builder. Admission still reads the
mode and records it in the invocation's EMPAIA input state, but nothing acts on
it yet. A preprocessing app, which EMPAIA starts with only a slide, thus runs
the heatmap builder over the whole slide.

EMPAIA apps are generated, not written by hand
(`src/model_service/providers/empaia/apps.py`). A model file opts in with an
`empaia` block: a display name (at most 22 characters, so `<name> <builder>`
fits EMPAIA's 30), a short name (at most 10), a description of what the model
finds, and the modes it runs in. Every builder that accepts the model's
contract and has an EMPAIA type for each output (today the heatmap builder,
not tile embeddings) becomes one app per listed mode: the standalone app takes
the slide and an optional region of interest, the preprocessing app only the
slide. Names, namespaces (`org.empaia.rationai.<builder>_<model>_<mode>.v3.0`),
the mode's sentence of the description, and the image
`http::<target>/builders/<builder>/models/<model>` are derived; each EAD is
validated against the bundled EAD v3 schema.

At startup the facade registers these apps in the marketplace set by
`EMPAIA_MARKETPLACE_URL` (Helm `empaia.marketplace`), in the background and
retrying every minute until it succeeds. It uses the marketplace mock's route
(`/v1/custom-mock/portal-apps`) that EMPAIA test installations run; a
production marketplace lists apps through its portal's review. App IDs derive
from the namespace, so registering again replaces an app, and every replica
may do it. Credentials for the marketplace go through an outgoing-credentials
interface: none by default, or client credentials from an identity provider
(`EMPAIA_IDP_URL`, `EMPAIA_CLIENT_ID`, `EMPAIA_CLIENT_SECRET`), obtained the way
`empaia-sender-auth` does.

EMPAIA slide inputs reach builders one of two ways, chosen by
`EMPAIA_SLIDE_ACCESS` (Helm `empaia.slideAccess`):

- `direct` (default): link the file named by the RationAI direct-access route
  (`/v3/rationai/{job}/slides/{input}/direct-access`). When the route has
  nothing for the input, the slide is downloaded as in `http`; a route that says
  direct access is unsupported is an error.
- `http`: download the slide through the App API's region route into the
  attempt workspace, for installations whose slide storage model-service cannot
  mount (the EMPAIA test instance's slides sit on a volume in another
  namespace). Only the level closest to the model's `mpp` is fetched, tile by
  tile and streamed into a tiled JPEG TIFF that records that level's
  resolution, so the builder reads it unchanged as level 0. Region-of-interest
  rectangles still become a local mask. The model's `mpp` reaches the input
  adapter with the invocation.
  When the job has a region of interest, only the tiles within one model tile
  of it are fetched (tiles the builder selects reach up to a tile past the
  region); the rest of the level is written white, which the builder treats as
  background. Jobs without one, such as preprocessing jobs, still fetch the
  whole level.

A running execution whose model or builder has no running Serve replica, only
one Ray has not placed on a node for `CAPACITY_GRACE_SECONDS` (default 60),
keeps `running` with `message: "waiting for capacity: no replica of model <id>
could be scheduled for <N>s"` (or `builder <id>`). Ray Serve keeps such a replica indefinitely and
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
- Due work is claimed by priority first, then as before: models with a waiting
  higher-priority execution are served first, and within a model higher
  priority goes first. Providers set it; the public API does not take it yet.
  An EMPAIA job in `STANDALONE` mode (someone started it in the Workbench and
  waits) is `HIGH`; preprocessing and everything else is `NORMAL`. The
  scale is `LOW`, `NORMAL`, `HIGH` (stored as 0, 1, 2).
  Priority only orders the queue: work already running keeps its slot.
  Migrations `0005` and `0006` add the column and its scale.
  Running these jobs outside the scheduler would not be faster: JES polls the
  execution every 5 s whichever way it runs, and the scheduler adds about a
  second (admission, the worker's 0.5 s poll, the claim) to jobs that take
  minutes. What a waiting user would notice is queueing behind preprocessing,
  which priority addresses, without losing durability across facade restarts.
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
- Request and response bytes are files, not database values, under
  `INFERENCE_STORAGE_PATH` (Helm `executionScheduler.inferenceStorage`: a
  ReadWriteMany claim mounted by every facade replica and nothing else, apart
  from the EMPAIA workspace so a burst of tiles cannot fill the volume
  publication needs). Files sit in 256 shard directories named by the first
  two characters of the execution ID, as `<id>.request` and
  `<id>.response-<fencing token>`; creating or deleting a file locks its
  directory, which on NFS serialized every tile when each execution had its
  own directory under one root. The row keeps each file's relative path and
  size, and a file whose size differs reads as missing. File I/O runs on its
  own 32-thread pool.
  - The request is written (temporary name, then rename) before the admission
    insert, so an acknowledged tile survives a restart. A replay or conflict
    removes the unused file. It is deleted once the execution finishes; a tile
    whose request file is gone fails with `inference_request_unavailable`.
  - Each attempt writes its response file before settling, and settlement
    records it only if the claim is still current; otherwise the file is
    removed, so a stale attempt never replaces the accepted response. A
    response that is gone reads as `409 result_unavailable`.
  - The response is kept for `INFERENCE_RETENTION_SECONDS` (24 h) after
    finishing; then the execution is deleted, and its files once that commits.
    `x-` response headers (e.g. `x-output-shape`) are kept in the row.
  - Every 10 minutes, maintenance removes execution files older than an hour
    that no row owns, left by a crash between writing a file and recording it.
  - Migration `0004` replaces the old `inference_payloads` table and removes
    existing inference executions, whose bytes cannot move to files; drain
    scheduled raw inference before upgrading.
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

EMPAIA wiring is one settings class, `EmpaiaSettings` (`EMPAIA_*`), read by
the facade and by builder replicas: `EMPAIA_API_URLS` (empty refuses every
EMPAIA job),
`EMPAIA_WORKSPACE_ROOT`, `EMPAIA_SLIDE_ACCESS`, and
`EMPAIA_APP_SERVICE_TIMEOUT_SECONDS` (default 60; creating a pixelmap took 10 s
on the test instance, and the SDK's 5 s default failed that job). Helm renders
them from the top-level `empaia` values for the facade and every worker group;
`empaia.dnsSearches` adds DNS search domains where an installation hands JES a
short App Service name from another namespace.

Attempt workspaces live under one directory per EMPAIA job
(`<workspace root>/<job key>/<attempt key>/`). Provider finish removes the
job's directory on every outcome: before a failure or cancellation report, and
after a successful publication (a failed publication keeps the outputs for its
retry). A failed input preparation removes its attempt directory at once.

EMPAIA is reached through the SDK's `rationai.empaia.Client` (the supervisor's
typed client with direct slide reading and pixelmap pyramid upload), built from
the SDK branch that adds raw job data (`get_configuration`, `get_mode`,
`get_input_data`), a request `timeout`, `set_before_mutation` (the scheduler's
write guard) and a public `get_direct_access_info`. The SDK retries region reads
and tile uploads; what it gives up on after creating a pixelmap has an unknown
outcome, so the finish step is not retried.

Heatmap publication. The heatmap builder keeps its raw scores: next to the TIFF
it writes `heatmap.tiff.scores.npz`, one score per cell, which cells were scored,
the cell's size in pixels of the level it read (256 for the prostate model) and
that level's size. It does so when the model has one channel and a cell covers
at least two pixels; a full-resolution output (segmentation) has no grid, and a
grid left in the place by an earlier build is removed. The EMPAIA publisher
publishes from the grid through the SDK's `ScalarImageOutput.to_pixelmap_pyramid`:
from the coarsest slide level that still keeps four pixels per score down to
the coarsest slide level, so every score is published as the model produced it,
without resampling the upsampled TIFF. A tile without a scored cell is not
uploaded; a skipped cell inside an uploaded tile reads 0, since pixelmap tiles
are dense. Without a grid, or with no scored cell, the publisher uploads the TIFF
as before: the coarsest level keeping four pixels per score, each pixel the area
average of the full-resolution heatmap.

### Safety and correctness fixes

- Shared-storage `root` must be one of the configured `SHARED_STORAGE_ROOTS`.
- EMPAIA `api_url` must match a configured `EMPAIA_API_URLS` prefix.
- A failed EMPAIA publication reports failure to EMPAIA instead of leaving the
  job open; idempotent pixelmap tile writes are retried.
- EMPAIA workspaces are removed once the job's outcome is reported, whatever it is.
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

## Operator tooling

Not part of the service, used to deploy and validate it:

- `scripts/warm_model.py`: one scheduled tile, to start a GPU replica before a
  demo or a job.
- `scripts/prepare_tissue_mask.py`: a tissue mask TIFF for standalone runs.
- `deploy/images/Dockerfile.code-refresh`: swaps the code layer of an existing
  image for test deployments; release images build from source.
- `testing/` doubles and `deploy/helm/rayservice/templates/validation-appservice-*`:
  the mock AppService and EMPAIA stand-ins.

## Out of scope

The EMPAIA bridge (JES's HTTP backend, registration, standalone and
preprocessing apps) is now in scope and implemented. Still out of scope:
annotation outputs, tile embeddings as EMPAIA outputs (EAD v3 has no type for
them), quality control (QC), concrete authentication and authorization
(only the interfaces exist), and batching tiles per model call. Automatic
preprocessing triggers are an EMPAIA admin's setting, not model-service's.
A tissue mask from a real source is future QC work: QC would run as a
preprocessing job whose mask later model-service jobs use; until then the
heatmap builder's own background filter stands in.
