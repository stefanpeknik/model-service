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
│  /v1/executions/{id}[/cancel|/result]   /v1/executions/lookup      │
│  /v1/empaia/builders/{b}/models/{m}  (JES HTTP execution backend)  │
│  model catalog ◄── ConfigMap from deploy/helm/rayservice/models/   │
│  execution scheduler (model_service/scheduler/) ◄──► PostgreSQL    │
└───────────────┬───────────────────────────────────┬────────────────┘
   builder call │ POST /builders/{b}/               │ raw tiles: POST /{m}/
┌───────────────▼───────────────────────────────────▼────────────────┐
│ Ray Serve (RayService)                                             │
│  builder-heatmap, builder-tile-embeddings (builders/replica/)    │
│        │ Ray handle or HTTP, per model contract                    │
│  model deployments: prostate-classifier-1, virchow2, …           │
└────────────────────────────────────────────────────────────────────┘
```

## Pieces

| Package | Owns |
| --- | --- |
| `src/model_service/models/implementations/` | Inference implementations (binary classifier, segmentation, Virchow2, Prov-GigaPath). Each serves any trained model that fits it; the trained model is chosen in YAML. |
| `deploy/helm/rayservice/models/` | One file per model deployment: an `inference` declaration plus the Ray Serve application. The same file feeds the RayService and the API's catalog. |
| `src/model_service/builders/` | The builder contract and runtime pieces (input access, atomic output files), shared slide reading (`slides/`) and one package per builder under `implementations/` (`heatmap/`, `embeddings/`: `spec.py`, `builder.py`, `processor.py`), listed in `registry.py`. See [Adding a builder](../guides/adding-builders.md). |
| `src/model_service/builders/replica/` | The one Ray Serve ingress (`serve.py`) that hosts any builder, and the runner that validates, verifies the model, enforces the deadline and cleans up. |
| `src/model_service/models/` | The model catalog (`catalog.py`), one model client per inference contract (`clients/`) and the MLflow loader (`artifacts.py`). |
| `src/model_service/api/` | The API process: public routes, synchronous runs, raw inference proxy, process wiring. |
| `src/model_service/api/http/` | Auth, HTTP error mapping and scheduling calls shared by the API routes and provider routes. |
| `src/model_service/contracts/` | Data shapes processes hand each other: invocation, result, failures, score grid, model declarations. |
| `src/model_service/scheduler/` | Durable executions: admission, leases with fencing, retries, deadlines, cancellation, provider finish. PostgreSQL with Yoyo migrations. |
| `src/model_service/providers/` | Providers: standalone (shared storage) and EMPAIA (AppService input and publication). `providers/registry.py` lists them; the API and the builder ingress compose what each contributes and name none of them. |

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

- One API process runs up to `MAX_CONCURRENT_EXECUTIONS` builder claims and,
  separately, `MAX_CONCURRENT_INFERENCE` tile claims. At most `PER_MODEL_LIMIT`
  builder attempts and `PER_MODEL_INFERENCE_LIMIT` tile attempts per model run
  at once, across all API replicas (PostgreSQL advisory lock per operation
  and model). Tiles are claimed in batches; one 1-CPU API replica handles
  about 100 tiles/s.
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
