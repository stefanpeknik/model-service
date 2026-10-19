# How model-service is put together

Model-service runs trained models and whole-slide **builders** on Ray Serve,
behind one public API that can also run builders durably.

```text
             callers: RationAI SDK, EMPAIA (via JES), notebooks
                                  │ HTTP
┌─────────────────────────────────▼──────────────────────────────────┐
│ facade (model_service/facade/, one Deployment, outside Ray)        │
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
│  builder-heatmap, builder-tile-embeddings (builders/serve.py)      │
│        │ Ray handle or HTTP, per model contract                    │
│  model deployments: prostate-classifier-1, virchow2, … (models/)   │
└────────────────────────────────────────────────────────────────────┘
```

## Pieces

| Package | Owns |
| --- | --- |
| `src/model_service/models/` | Inference implementations (binary classifier, segmentation, Virchow2, Prov-GigaPath). Each serves any trained model that fits it; the trained model is chosen in YAML. |
| `deploy/helm/rayservice/models/` | One file per model deployment: an `inference` declaration plus the Ray Serve application. The same file feeds the RayService and the facade's catalog. |
| `src/model_service/builders/` | Builder registry, the builder runtime (validation, deadline, input access, cleanup) and the two builders. One Ray ingress, `src/model_service/builders/serve.py`, hosts any builder. |
| `src/model_service/facade/` | The facade: public API, model catalog, synchronous runs, raw inference proxy, process wiring. |
| `src/model_service/scheduler/` | Durable executions: admission, leases with fencing, retries, deadlines, cancellation, provider finish. PostgreSQL with Yoyo migrations. |
| `src/model_service/providers/` | Providers: standalone (shared storage) and EMPAIA (AppService input and publication). |

## A request, end to end

1. The caller names a builder in the URL and a model in the body.
2. The facade checks both exist and that the model's inference contract is the
   one the builder accepts, then checks inputs.
3. Synchronous: it forwards the invocation plus the model's current catalog
   entry to `/builders/{b}/`. Scheduled: it stores the invocation (not the
   model entry) and the worker resolves the model again on every attempt, so
   queued work always uses current settings.
4. The builder replica opens inputs through an access adapter, runs, and writes
   its artifact atomically. Scheduled attempts write under
   `.attempts/<hash of attempt token>/` so a stale attempt never overwrites a
   newer one.
5. Provider finish delivers the result: nothing to do for standalone, or
   publication to the AppService for EMPAIA.

A scheduled tile (`POST /v1/models/{m}/schedule`) is the same lifecycle with a
different payload: the facade checks the tile against the model's `tile_size`,
stores its bytes with the execution, and an attempt sends them to the
model's Ray route exactly as `POST /v1/models/{m}/run` would. The response is stored for a
day and served by `GET /v1/executions/{id}/result`. There is no provider
finish. `POST /v1/executions/lookup` polls many executions at once.

## Contracts

| Contract | Models | How a builder calls it | Declared where |
| --- | --- | --- | --- |
| `spatial-scores` | binary classifier, semantic segmentation, constant test model | Ray handle `predict(CHW uint8)`; the heatmap builder checks `get_config()` against the declaration before starting | `tile_size`, `output_tile_size`, `n_channels`, `mpp` in `user_config` |
| `tile-embedding` | Virchow2, Prov-GigaPath | Ray handle `embed(CHW uint8)` → pooled features; the builder checks `get_config()` (`tile_size`, `summaries`) first | `tile_size` in `user_config`; `mpp`, `summaries` in `inference` |

## Capacity and failure behaviour

- One facade process runs up to `MAX_CONCURRENT_EXECUTIONS` builder claims and,
  separately, `MAX_CONCURRENT_INFERENCE` tile claims. At most `PER_MODEL_LIMIT`
  builder attempts and `PER_MODEL_INFERENCE_LIMIT` tile attempts per model run
  at once, across all facade replicas (PostgreSQL advisory lock per operation
  and model). Tiles are claimed in batches; one 1-CPU facade replica handles
  about 100 tiles/s.
- Builder replicas return 429 when full; the scheduler requeues without
  spending an attempt. Synchronous callers get 429 with `Retry-After`.
- Delivery is at least once. A lost facade lets its lease expire and another
  attempt runs. Lease renewal survives short database outages.
- A failed or uncertain EMPAIA publication reports failure to EMPAIA, so the
  job does not stay open.

The scheduler runs outside Ray Serve so admission, polling and cancellation keep
working while compute is down; see
[ADR 0001](../adr/0001-run-scheduler-outside-ray-serve.md).
