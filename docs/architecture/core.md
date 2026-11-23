# How model-service is put together

Model-service runs trained models (**served models**) and whole-slide
**analyses** on Ray Serve, behind one public API that can also run analyses
durably. Both are YAML files in `catalog/` that name **components**, small
classes under `src/components/`. Those two folders are what a newcomer
writes; `src/model_service/` is the plumbing that loads, checks and runs
them. The
[guides](../guides/add-a-model.md) show how to add each, and the
[reference](../reference/components.md) lists what exists.

```text
             callers: RationAI SDK, EMPAIA (via JES), notebooks
                                  │ HTTP                           │ raw tiles
┌─────────────────────────────────▼────────────────────────────┐   │
│ API process (model_service/api/, one Deployment, outside Ray)│   │
│  /v1/models[/{m}]   (served-model catalog)                   │   │
│  /v1/analyses   /v1/analyses/{a}/run|schedule                │   │
│  /v1/executions/{id}[/cancel]   /v1/executions/lookup        │   │
│  /v1/empaia/analyses/{a}  (JES HTTP execution backend)       │   │
│  catalog ◄── ConfigMap rendered from catalog/                │   │
│  execution scheduler (api/scheduler/) ◄──► PostgreSQL        │   │
└───────────────┬──────────────────────────────────────────────┘   │
      run call  │ POST /analyses/                                  │ POST /{m}/
┌───────────────▼──────────────────────────────────────────────────▼─┐
│ Ray Serve (RayService)                                             │
│  analysis-runner (analysis_runner/serve.py), POST /analyses/       │
│        │ Ray handle: infer(tile), describe()                       │
│  served models (served_model/serve.py): prostate-classifier-1, …   │
└────────────────────────────────────────────────────────────────────┘
```

## Pieces

| Package | Owns |
| --- | --- |
| `deploy/helm/rayservice/` (catalog wiring) | The chart reads the repo's `catalog/` through its `catalog` symlink (`helm template` and `helm package` follow it). Every `catalog/served-models/<id>.yaml` that the values' `servedModels` selects (`["*"]` or named IDs) becomes one generic-host Serve application, with `catalog/environments/<environment>/` patches deep-merged in and environment-wide settings from `servedModelSettings` (the cache root, the replicas' env vars and the applications' `runtime_env` setup timeout); `deployment.serve_application` is the reference, and `tests/deploy/test_chart.py` compares the rendered applications with it. `catalog/analyses/` is selected the same way by `analyses`; an analysis calling a served model the release doesn't deploy fails the render, and with any analysis deployed the chart renders one `analysis-runner` application sized by `analysisRunner`. The analysis runner requests a `cpuWorkerResource` Ray custom resource (`cpu_worker`) that only the CPU worker groups advertise in `rayStartParams`, so Ray never places it on a GPU worker's free CPUs; served models don't request it. The catalog ConfigMap's `catalog.json` carries both folders; the prostate app is `prostate-heatmap`'s `expose.empaia`. |
| `src/model_service/api/` | The API process: public routes, synchronous runs, process wiring. The catalog ConfigMap is mounted whole at `MODEL_CATALOG_DIR`; `catalog_files.py` resolves its `catalog.json` through `load_catalog`, and `catalog_mount.py` reloads it when kubelet updates the files, keeping the old catalog when the new one fails and reporting why under `/healthz/catalog`. A catalog change no longer restarts the API. |
| `src/model_service/api/http/` | Auth, HTTP error mapping and scheduling calls shared by the API routes and provider routes. |
| `src/model_service/contracts/` | Data shapes processes hand each other: invocation, result, failures, the analysis runner's HTTP request and error envelope. |
| `src/model_service/api/scheduler/` | Durable executions: admission, leases with fencing, retries, deadlines, cancellation, provider finish. PostgreSQL with Yoyo migrations. |
| `src/model_service/providers/` | Providers: standalone (shared storage) and EMPAIA (AppService input and publication). `providers/registry.py` lists them; the API and the analysis runner compose what each contributes and name none of them. `providers/input_access.py` is the interface their input access implements. |
| `src/components/` | A top-level package beside `model_service`, holding what an author writes. `base/` has the seven kind base classes (loader, runtime, step, source, sink, input converter, publisher), options with the `PerRun` marker, column declarations, `Run`, `Problem` and the slots; it imports nothing from `model_service` at run time. The components sit in one folder per place of use: `steps/` (served model `pre:`/`post:` and analysis steps), `loaders/` and `runtimes/` (served models; a loader also fills an analysis step's `weights:`), `sources/` and `sinks/` (analyses), `empaia/converters/` and `empaia/publishers/` (EMPAIA exposure). `_templates/` has one file per kind to copy; the scan skips it. The helpers for testing a component (`fake_slide`, `fake_model`, `run_component`, `FakeGateway`, the per-kind conformance suites and `offline.py`, the table of local weights that lets conformance run a loader or runtime by name) are test code in `tests/components/helpers/`, so images don't ship them. Analysis components so far: the `slide_tiles` source (`slides/`'s `TileReader` and ratiopath's grid) the `score_grid` sink (ratiopath's `MaskBuilder`) and the `embedding_table` sink (rows in grid order, written on a thread of its own). |
| `src/model_service/registry/` | Finds and checks the components. Each kind's base class names its `folder` beside its `slots`, so the kind alone says where its components live and where catalog files may name them. `Registry.scan()` imports each kind's folder and fails when a component sits in another kind's folder (the error names the folder it belongs in) or two of one kind share a name; `default_registry()` is the scan of `components`, once per process. `registry/empaia.py` picks the converter or publisher of a declared EAD type. The rules sit here rather than in `components/` so that package holds only the pieces and what they subclass. |
| `src/model_service/catalog/` and `catalog/` | `load_catalog(root, environment)` deep-merges `catalog/environments/<env>/` patches over the base files and resolves the served models, then the analyses, into a typed `Catalog` (`served_models` and `analyses` by ID, each keeping its patched `declaration`), reporting every mistake as `<file>:<yaml path>: <what's wrong>. <what fits>. Hint: <fix>`. `tests/catalog/` loads the repo's `catalog/` for every environment. Served-model files (`catalog/served-models/`) resolve to a loader, a runtime, `pre`/`post` steps and the `input`, `output`, `batching`, `resources` and `serving` blocks, with a digest of what the model computes. Analysis files (`catalog/analyses/`) resolve to a source, steps (`{model: <id>}` calls a served model) and sinks; loading fills the source's tile geometry from the first model step (a source opts in by deriving its options from `TileGeometryOptions`), folds the columns from source to sinks with the models' dimensions, checks the frame fields sinks read and collects the `PerRun` options as flat settings. `prostate-heatmap.yaml` and `episeg-heatmap.yaml` score tiles into a heatmap (EPISeg's 1024 px float16 score map per tile is averaged into a grid at the read level's resolution; the test cluster doesn't deploy it); `virchow2-embeddings.yaml` and `prov-gigapath-embeddings.yaml` write one embedding row per tissue tile. |
| `src/model_service/slides/` | The team's slide reading for analysis sources: `TileReader` picks the level closest to a resolution and drops tiles outside the tissue or the region of interest; `tissue.py` detects tissue with ratiopath. |
| `src/model_service/analysis_runner/` | `run_analysis(analysis, inputs, outputs, models)`, a plain async function that runs any resolved analysis with 2 read threads and 4 tiles in flight, progress per source key, per-step timing, failure codes, ordered delivery when a sink needs it, and all-or-nothing writes across sinks. On the test cluster, 3 of the 4 analyses (prostate-heatmap, virchow2-embeddings, prov-gigapath-embeddings) match the golden references recorded on 2026-11-22 within their tolerances ([record](../operations/golden-references-2026-11-22.md)); episeg-heatmap has none. `AnalysisReplica` (`replica.py`) runs one request: it reads back the resolved document and its digest, compares each served model's `describe()` digest with it (`model_contract_mismatch` on a difference), resolves inputs and outputs below an allowed shared-storage root and runs before the deadline. `serve.py` is its Serve wrapper, deployed as the one `analysis-runner` application under `/analyses`. |
| `src/model_service/formats/` | Data formats sinks write and readers open: one light module each with a versioned vendor media type, a dataclass, `read`/`write`/`check`. `score_grid` v1 is the heatmap's npz layout (`application/vnd.rationai.score-grid+npz; version=1`); `embedding_table` v2 is the tile-embeddings HDF5 layout (`application/vnd.rationai.embedding-table+hdf5; version=2`, the file's own `schema_version`), with a `TableWriter` that appends rows. |
| `src/model_service/served_model/` | The generic host that serves every served model. `ServedModelHost` (a plain class) batches, guards inputs and outputs against the declaration, answers the raw route and reports `describe()`, which it also logs when built. The raw route's unpooled reply (`x-pool-tokens: false`) skips `post:` and drops the register tokens a `cls_mean_pool` post step names (`replies.unpooled`), so steps carry no transport concern. `serve.py` is its Serve wrapper, which serves `describe()` as `GET /<id>/describe` too (`model_service.served_model.serve:app`); `deployment.py` defines the Serve application each served model deploys as. `prostate-classifier-1`, `virchow2` (the `huggingface` loader, the `timm` runtime and the `cls_mean_pool` post step) and `prov-gigapath` (the `timm` runtime with `resize`, `center_crop` and `normalize` pre steps, which crop a 224 px tile to its centre as the recorded references do) match their golden references on the test cluster within their tolerances. `episeg-1` has no live check and no live references, since its weights are out of the test cluster's reach; it keeps ONNX Runtime's default session options (`apply_session_options: false`), as its old class ran. The chart deploys every served model on it; the test cluster deploys `prostate-classifier-1`, `virchow2` and `prov-gigapath` (EPISeg's weights are unreachable there), the base values all four. |
| `src/model_service/reference/`, `tools/generate.py` | The generated files: the served-model and analysis JSON Schemas in `catalog/.schemas/` (with a patch variant that requires nothing), the `yaml-language-server` line at the top of each catalog file, and the pages in `docs/reference/`. They come from the component classes, the format modules and `catalog/`; `tests/test_generated.py` fails when a committed copy is stale. |

### Interfaces

Interfaces between parts are abstract base classes (`ABC` with
`@abstractmethod`), and every implementation inherits its interface:
`InputAccess`, `InputAccessFactory`, `ProviderAdapter`,
`ExecutionRepository`, `EmpaiaGateway` and the rest. A class missing a method
then fails when it is created, not when the method is first called. Test fakes
need not inherit them. Two are `Protocol`s instead: the API's `Scheduler`,
because the core scheduler must not import the API, and
`EmpaiaGatewayFactory`, which a plain function implements.

### Providers

A provider is a package that plugs into the API and the analysis runner
through one object (`Provider` in `providers/provider.py`, listed in
`providers/registry.py`):

- `api(catalog, settings)` gives the scheduler adapter that admits and
  finishes its jobs, any routes of its own (EMPAIA's `/v1/empaia`), and work
  to run in the background (EMPAIA's app registration);
- `input_access(settings)` gives the analysis runner the access adapters for
  the inputs its jobs hand out.

Providers name themselves and their input access kinds; the runner, the
catalog and the scheduler know none of them. How an analysis appears in EMPAIA
is its `expose.empaia` section, made into an app from the converters' and
publishers' fragments (`providers/empaia/ead/apps.py`). The API's public
analysis routes are the standalone provider's front door and build on its
request body. Adding a provider means a package and one entry in `PROVIDERS`
(`providers/registry.py`).

## A request, end to end

1. The caller names an analysis in the URL; the body gives inputs, their
   access, settings (`configuration`) and an output path.
2. The API resolves the analysis with the settings and checks the inputs.
3. Synchronous: it sends the resolved document and the invocation to the
   analysis runner (`/analyses/`). Scheduled: it stores both on the execution,
   and every attempt sends that document.
4. The runner checks each served model's `describe()` digest against the
   document, and its weights revision where the served-model file pins one
   (see below), opens inputs through an access adapter, runs, and commits its
   outputs all or nothing. Scheduled attempts write under
   `.attempts/<execution>/<hash of attempt token>/` so a stale attempt never
   overwrites a newer one.
5. Provider finish delivers the result. Standalone moves the accepted
   artifacts to their requested paths, stores those paths in the result and
   removes the execution's attempt directories; EMPAIA publishes to the
   AppService.

### Which weights a served model runs

The digest covers the served-model file, so a replica deployed from another
file is refused. The file names its weights, but only some names fix the
bytes:

- `huggingface` pins a commit, so the runner compares `describe()`'s
  `weights_revision` (`<repo>@<commit>`) with the file's.
- `mlflow` names a run's artifact by URI, and an artifact can be overwritten
  under the same URI. The loader checksums what it downloaded, so
  `describe()` and the replica's start-up log show
  `<uri>@sha256:<hex>`. With `sha256:` in the file, a download with other
  bytes fails the replica's start and the runner compares the revision too.
  Without it, which the catalog's MLflow models have today, nothing checks
  the bytes: a changed artifact is visible in `describe()` but runs.
- `local` names a path; the bytes at it are not checked.

A scheduled analysis (`POST /v1/analyses/{a}/schedule`, the synchronous
run's body with `requested_timeout_seconds` instead of a deadline and an
`Idempotency-Key` header) is resolved once, at admission: the scheduler
resolves the analysis with the caller's settings, checks the inputs and
stores the resolved document and its digest on the execution. Every attempt,
retries included, sends that stored document to the analysis runner, so a
catalog change after admission doesn't change what runs; the runner's
`describe()` check turns a changed served model into `model_contract_mismatch`.
An execution's identity is its analysis ID, the analysis digest and the
served models it calls. `GET /v1/models` lists each served
model's declaration, digest and the analyses calling it.

Raw tiles don't pass through the API: callers post them straight to the
served model's Ray Serve route, `POST /{m}/`
([raw inference](../available-models.md)). `POST /v1/executions/lookup` polls
many executions at once.

## Capacity and failure behaviour

- One API process runs up to `MAX_CONCURRENT_EXECUTIONS` claims. A scheduled
  attempt takes a slot in every served model its analysis calls, and is
  claimable only while each has one free, across all API replicas. A model's
  slots are its file's `serving.max_concurrent_runs`, or `PER_MODEL_LIMIT`
  (default 1) when left out. An analysis calling no model has its own pool,
  keyed by its ID, with `PER_MODEL_LIMIT` slots. Claims take one PostgreSQL
  advisory lock per pool, tried in sorted order, so overlapping claims
  never wait on each other.
- The analysis runner returns 429 when full; the scheduler requeues without
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

## Known limits

- A deploy interrupts the runs in flight. A scheduled run's lease then
  expires and another attempt starts from the beginning; a synchronous
  caller gets an error and has to call again.
- A model redeployed in the middle of a run can answer some of its tiles
  with the old version and some with the new one. The digest check happens
  once, when the run starts.
- A model call has no timeout of its own. The run's deadline is the only
  bound, so a stuck replica holds the run until the deadline passes.
- One tile that fails fails the whole run, and the retry starts again from
  the first tile.
- The analysis runner trusts everything that reaches it on the cluster
  network. Authentication and hardening between processes are a separate
  piece of work.
- Steps can't load weights of their own yet; only a served model's loader
  fetches weights.
