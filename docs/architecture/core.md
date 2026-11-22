# How model-service is put together

Model-service runs trained models and whole-slide **builders** on Ray Serve,
behind one public API that can also run builders durably.

```text
             callers: RationAI SDK, EMPAIA (via JES), notebooks
                                  │ HTTP
┌─────────────────────────────────▼──────────────────────────────────┐
│ API process (model_service/api/, one Deployment, outside Ray)      │
│  /v1/models   /v1/models/{m}/run|schedule (raw tiles)              │
│  /v1/builders   /v1/builders/{b}/models/{m}/run|schedule           │
│  /v1/analyses   /v1/analyses/{a}/run|schedule                      │
│  /v1/executions/{id}[/cancel|/result]   /v1/executions/lookup      │
│  /v1/empaia/builders/{b}/models/{m}  (JES HTTP execution backend)  │
│  model catalog ◄── ConfigMap from deploy/helm/rayservice/models/   │
│  execution scheduler (model_service/api/scheduler/) ◄──► PostgreSQL    │
└───────────────┬───────────────────────────────────┬────────────────┘
   builder call │ POST /builders/{b}/               │ raw tiles: POST /{m}/
┌───────────────▼───────────────────────────────────▼────────────────┐
│ Ray Serve (RayService)                                             │
│  builder-heatmap, builder-tile-embeddings (builders/replica/)    │
│  analysis-runner (analysis_runner/serve.py), POST /analyses/       │
│        │ Ray handle or HTTP, per model contract                    │
│  model deployments: prostate-classifier-1, virchow2, …           │
└────────────────────────────────────────────────────────────────────┘
```

## Pieces

| Package | Owns |
| --- | --- |
| `src/model_service/inference/implementations/` | Inference implementations (binary classifier, segmentation, Virchow2, Prov-GigaPath). Each serves any trained model that fits it; the trained model is chosen in YAML. |
| `deploy/helm/rayservice/models/` | Models not yet migrated to `catalog/served-models/` (transitional): one file per model deployment, a contract plus the Ray Serve application, listed under `models:` in the values file. The same file feeds the RayService and the API's catalog. |
| `deploy/helm/rayservice/` (catalog wiring) | The chart reads the repo's `catalog/` through its `catalog` symlink (`helm template` and `helm package` follow it). Every `catalog/served-models/<id>.yaml` that the values' `servedModels` selects (`["*"]` or named IDs) becomes one generic-host Serve application, with `catalog/environments/<environment>/` patches deep-merged in and environment-wide settings from `servedModelSettings` (the cache root, the replicas' env vars and the applications' `runtime_env` setup timeout); `deployment.serve_application` is the reference, and `tests/deploy/test_chart.py` compares the rendered applications with it. `catalog/analyses/` is selected the same way by `analyses`; an analysis calling a served model the release doesn't deploy fails the render, and with any analysis deployed the chart renders one `analysis-runner` application sized by `analysisRunner`. Builders and the analysis runner request a `cpuWorkerResource` Ray custom resource (`cpu_worker`) that only the CPU worker groups advertise in `rayStartParams`, so Ray never places them on a GPU worker's free CPUs; served models don't request it. The catalog ConfigMap's `catalog.json` carries both folders. `provider-blocks/<id>.yaml` holds a served model's provider blocks (the prostate `empaia` block) until analyses carry them. |
| `src/model_service/builders/` | The builder contract and runtime pieces (input access, atomic output files), shared slide reading (`slides/`), one model client per inference contract (`model_clients/`) and one package per builder under `implementations/` (`heatmap/`, `embeddings/`: `spec.py`, `builder.py`, `processor.py`), listed in `registry.py`. See [Adding a builder](../guides/adding-builders.md). |
| `src/model_service/builders/replica/` | The one Ray Serve ingress (`serve.py`) that hosts any builder, and the runner that validates, verifies the model, enforces the deadline and cleans up. |
| `src/model_service/inference/` | The MLflow loader (`artifacts.py`) the model files name as `_target_`. |
| `src/model_service/api/` | The API process: public routes, synchronous runs, the model catalog (`model_catalog.py`), process wiring. The catalog ConfigMap is mounted whole at `MODEL_CATALOG_DIR`; `catalog_files.py` resolves its served models through `load_catalog` and derives today's bindings from them beside the old model files, and `catalog_mount.py` reloads it when kubelet updates the files, keeping the old catalog when the new one fails and reporting why under `/healthz/catalog`. A catalog change no longer restarts the API. |
| `src/model_service/api/http/` | Auth, HTTP error mapping and scheduling calls shared by the API routes and provider routes. |
| `src/model_service/contracts/` | Data shapes processes hand each other: invocation, result, failures, score grid, model declarations. |
| `src/model_service/api/scheduler/` | Durable executions: admission, leases with fencing, retries, deadlines, cancellation, provider finish. PostgreSQL with Yoyo migrations. |
| `src/model_service/providers/` | Providers: standalone (shared storage) and EMPAIA (AppService input and publication). `providers/registry.py` lists them; the API and the builder ingress compose what each contributes and name none of them. |
| `src/model_service/components/` | Being built, not yet wired in: the seven component kinds (loader, runtime, step, source, sink, input converter, publisher), options with the `PerRun` marker, column declarations, and the registry that scans the component folders (`served_model/`, `steps/`, `analysis/`, `empaia/`, `standalone/`). Analysis components so far: the `slide_tiles` source (the builder's `TileReader` and ratiopath's grid) the `score_grid` sink (ratiopath's `MaskBuilder`) and the `embedding_table` sink (rows in grid order, written on a thread of its own). |
| `src/model_service/catalog/` and `catalog/` | `load_catalog(root, environment)` deep-merges `catalog/environments/<env>/` patches over the base files and resolves each document kind, reporting every mistake as `<file>:<yaml path>: <what's wrong>. <what fits>. Hint: <fix>`. `tests/catalog/` loads the repo's `catalog/` for every environment. Served-model files (`catalog/served-models/`) resolve to a loader, a runtime, `pre`/`post` steps and the `input`, `output`, `batching`, `resources` and `serving` blocks, with a digest of what the model computes. Analysis files (`catalog/analyses/`) resolve to a source, steps (`{model: <id>}` calls a served model) and sinks; loading fills the source's tile geometry from the first model step, folds the columns from source to sinks with the models' dimensions, checks the frame fields sinks read and collects the `PerRun` options as flat settings. `prostate-heatmap.yaml` and `episeg-heatmap.yaml` are today's heatmap builder on prostate and on EPISeg (EPISeg's 1024 px float16 score map per tile is averaged into a grid at the read level's resolution; the test cluster doesn't deploy it); `virchow2-embeddings.yaml` and `prov-gigapath-embeddings.yaml` are today's tile-embeddings builder on each encoder. |
| `src/model_service/analysis_runner/` | `run_analysis(analysis, inputs, outputs, models)`, a plain async function that runs any resolved analysis with today's concurrency (2 read threads, 4 tiles in flight), progress per source key, per-step timing, failure codes, ordered delivery when a sink needs it, and all-or-nothing writes across sinks. On a fake slide its `prostate-heatmap` and `episeg-heatmap` score grids are byte for byte today's heatmap processor's with one tile in flight, and the embeddings analyses' HDF5 files are byte for byte today's embeddings processor's at today's concurrency. `AnalysisReplica` (`replica.py`) runs one request: it reads back the resolved document and its digest, compares each served model's `describe()` digest with it (`model_contract_mismatch` on a difference), resolves inputs and outputs below an allowed shared-storage root and runs before the deadline. `serve.py` is its Serve wrapper, deployed as the one `analysis-runner` application under `/analyses`. |
| `src/model_service/formats/` | Data formats sinks write and readers open: one light module each with a versioned vendor media type, a dataclass, `read`/`write`/`check`. `score_grid` v1 is the heatmap's npz layout (`application/vnd.rationai.score-grid+npz; version=1`); `embedding_table` v2 is the tile-embeddings HDF5 layout (`application/vnd.rationai.embedding-table+hdf5; version=2`, the file's own `schema_version`), with a `TableWriter` that appends rows; `contracts/score_grid.py` names it for the builder until the builders retire. |
| `src/model_service/served_model/` | The generic host that will serve every served model. `ServedModelHost` (a plain class) batches, guards inputs and outputs against the declaration, answers the raw route in today's formats and reports `describe()`; `serve.py` is its Serve wrapper (`model_service.served_model.serve:app`); `deployment.py` defines the Serve application each served model deploys as. `prostate-classifier-1` is bit-identical to `BinaryClassifier`, `episeg-1` to `SemanticSegmentation`, `virchow2` (the `huggingface` loader, the `timm` runtime and the `cls_mean_pool` post step) to `Virchow2` and `prov-gigapath` (the `timm` runtime with `resize`, `center_crop` and `normalize` pre steps, keeping today's crop of a 224 px tile to its centre) to `ProvGigaPath`, on stand-in weights; `episeg-1` keeps ONNX Runtime's default session options (`apply_session_options: false`) because the old class never applied its own. The chart deploys every migrated served model on it; the test cluster deploys `prostate-classifier-1`, `virchow2` and `prov-gigapath` (EPISeg's weights are unreachable there), the base values all four. |
| `src/model_service/testing/` | `fake_slide`, `fake_model`, `run_component` and the per-kind conformance suites that run for every registered component. |

### Interfaces

Interfaces between parts are abstract base classes (`ABC` with
`@abstractmethod`), and every implementation inherits its interface:
`Builder`, `AccessAdapter`, `ProviderAdapter`, `OutputPublisher`,
`ExecutionRepository`, `EmpaiaGateway` and the rest. A class missing a method
then fails when it is created, not when the method is first called. Test fakes
need not inherit them. Two are `Protocol`s instead: the API's `Scheduler`,
because the core scheduler must not import the API, and
`EmpaiaGatewayFactory`, which a plain function implements.

### Providers

A provider is a package that plugs into the API and the builder replicas
through one object (`Provider` in `providers/registry.py`):

- `api(catalog, settings)` gives the scheduler adapter that admits and
  finishes its jobs, any routes of its own (EMPAIA's `/v1/empaia`), and work
  to run in the background (EMPAIA's app registration);
- `catalog_block` names the top-level block of the model files it reads (the
  catalog keeps it unread and rejects blocks no provider takes);
- `input_access(settings)` gives builder replicas the access adapters for the
  inputs its jobs hand out.

Providers name themselves and their input access kinds; the builders, the
catalog and the scheduler know none of them. How builders appear in EMPAIA, the
EAD `io` of their inputs and outputs, lives in `providers/empaia/io.py`. The
API's public builder routes are the standalone provider's front door and
build on its request body. Adding a provider means a package and one entry in
`PROVIDERS` (`providers/installed.py`).

## A request, end to end

1. The caller names a builder in the URL and a model in the body.
2. The API checks both exist and that the model's inference contract is the
   one the builder accepts, then checks inputs.
3. Synchronous: it forwards the invocation plus the model's current catalog
   entry to `/builders/{b}/`. Scheduled: it stores the invocation (not the
   model entry) and the worker resolves the model again on every attempt, so
   queued work always uses current settings.
4. The builder replica opens inputs through an access adapter, runs, and writes
   its artifact atomically. Scheduled attempts write under
   `.attempts/<execution>/<hash of attempt token>/` so a stale attempt never
   overwrites a newer one.
5. Provider finish delivers the result. Standalone moves the accepted
   artifacts to their requested paths, stores those paths in the result and
   removes the execution's attempt directories; EMPAIA publishes to the
   AppService.

A scheduled analysis (`POST /v1/analyses/{a}/schedule`, the synchronous
run's body with `requested_timeout_seconds` instead of a deadline and an
`Idempotency-Key` header) is resolved once, at admission: the scheduler
resolves the analysis with the caller's settings, checks the inputs and
stores the resolved document and its digest on the execution. Every attempt,
retries included, sends that stored document to the analysis runner, so a
catalog change after admission doesn't change what runs; the runner's
`describe()` check turns a changed served model into `model_contract_mismatch`.
An execution's identity is its analysis ID, the analysis digest and the
served models it calls (migration `0009`). Builder runs (the builder routes
and EMPAIA, until its app moves to analyses) store no document: their
analysis ID is the day-one analysis of their builder and model
(`heatmap` × `prostate-classifier-1` is `prostate-heatmap`, and so on, see
`api/scheduler/targets.py`) or `<builder>:<model>`, and each attempt runs
that builder with the model's current catalog entry. `GET /v1/models` lists
each served model's declaration, digest and the analyses calling it.

A scheduled tile (`POST /v1/models/{m}/schedule`) is the same lifecycle with a
different payload: the API checks the tile against the model's `tile_size`,
writes its bytes to a file on the inference storage volume that the execution
row points to, and an attempt sends them to the
model's Ray route exactly as `POST /v1/models/{m}/run` would. The response is stored for a
day and served by `GET /v1/executions/{id}/result`. There is no provider
finish. `POST /v1/executions/lookup` polls many executions at once.

## Contracts

| Contract | Models | How a builder calls it | Declared where |
| --- | --- | --- | --- |
| `spatial-scores` | binary classifier, semantic segmentation, constant test model | `SpatialScoresClient`: Ray handle `predict(CHW uint8)` → `n_channels x output_tile_size²` finite scores | `tile_size`, `output_tile_size`, `n_channels`, `mpp` in `user_config` |
| `tile-embedding` | Virchow2, Prov-GigaPath | `TileEmbeddingClient`: Ray handle `embed(CHW uint8)` → finite float32 features per summary | `tile_size` in `user_config`; `mpp`, `summaries` in `inference` |

Before a builder runs, the application checks the running model's
`get_config()` against its catalog entry (`model_contract_mismatch` if they
differ). Every builder of a contract then calls the model through the same
client: a failed call is `model_failed` (retried), and an answer outside the
contract is `invalid_model_output` (not retried).

## Capacity and failure behaviour

- One API process runs up to `MAX_CONCURRENT_EXECUTIONS` claims. A scheduled
  attempt takes a slot in every served model its analysis calls, and is
  claimable only while each has one free, across all API replicas. A model's
  slots are its file's `serving.max_concurrent_runs`, or `PER_MODEL_LIMIT`
  (default 1) when left out. An analysis calling no model has its own pool,
  keyed by its ID, with `PER_MODEL_LIMIT` slots. Claims take one PostgreSQL
  advisory lock per pool, tried in sorted order, so overlapping claims
  never wait on each other.
- Builder replicas return 429 when full; the scheduler requeues without
  spending an attempt. Ray Serve's own non-JSON 503 (replica queue full, none
  running) counts the same. Synchronous callers get 429 with `Retry-After`.
- Only a claim decrypts an execution's provider context, and it is erased once
  no work is left. A context that cannot be decrypted ends only its execution
  (`protected_context_unreadable`, or its outcome if it already had one, with
  no report) and its stored bytes are kept.
- An EMPAIA 429 on a write is a refusal, so it may be retried; a 5xx may have
  applied, so its outcome is unknown. The deadline applies to publication too:
  one that ends past it is recorded as outcome unknown.
- Delivery is at least once. A lost API lets its lease expire and another
  attempt runs. Lease renewal survives short database outages.
- A failed or uncertain EMPAIA publication reports failure to EMPAIA, so the
  job does not stay open.

The scheduler runs outside Ray Serve so admission, polling and cancellation keep
working while compute is down; see
[ADR 0001](../adr/0001-run-scheduler-outside-ray-serve.md).
