# What this thesis added to model-service

The starting point is the team's model-service at commit `3d37a6e` ("Fix:
issues (#9)"): Ray Serve inference implementations for a binary classifier,
semantic segmentation, Virchow2 and Prov-GigaPath, a heatmap builder script and
Helm charts. Callers reached each model through the Ray HTTP proxy.

What the department can do now that it could not before: queue many slides
to run overnight as GPUs free up, offer models to researchers as EMPAIA apps,
and add a trained model with one YAML file.

248 commits later the source and tests have grown by about 29,700 lines across
184 files (`git diff --shortstat 3d37a6e HEAD -- src tests`).

## Added

| Area | What it gives | Where |
| --- | --- | --- |
| Public API | One HTTP entry point for models, builders and executions, with auth hooks, capacity checks and health | `src/model_service/api/`, `http_common/` |
| Durable scheduler | Executions stored in PostgreSQL: queueing until capacity frees, leases, retries, deadlines, cancellation, priorities | `src/model_service/scheduler/` |
| Builders | Whole-slide outputs as plug-ins: heatmap (score grid) and tile embeddings (HDF5), one Ray Serve app hosting any builder | `builders/`, `builder_deployment/` |
| Model catalog | One YAML per trained model feeds both Ray Serve and the API; the API refuses to start on an inconsistent file, and each run checks the running model matches its declaration | `inference/catalog.py`, `deploy/helm/rayservice/models/` |
| EMPAIA provider | Self-registration as an EMPAIA app, JES job execution in STANDALONE and PREPROCESSING modes, publication of primitives and pixelmaps through the EMPAIA SDK | `providers/empaia/`, [EMPAIA integration](architecture/empaia.md) |
| Tests | About 380 unit and PostgreSQL tests, including a contract test kit every builder runs | `tests/` |

## Preserved

The inference implementations still do the same maths; they moved from
`src/model_service/models/` to `src/model_service/model_implementations/`.

- `semantic_segmentation.py`: unchanged apart from imports.
- `virchow2.py`, `prov_gigapath.py`: the HTTP handler was split into
  `_encode`, `_pool` and `embed` so builders can call the model by Ray handle.
  The embedding is unchanged (class token plus the mean of patch tokens), and the
  LZ4 HTTP route remains.
- `binary_classifier.py`: adds `get_config()` and the geometry fields
  (`output_tile_size`, `n_channels`, `mpp`) the catalog checks. It also passes
  ONNX Runtime options as `sess_options=`, its real keyword; the baseline's
  `session_options=` was silently ignored, so the TensorRT settings now take
  effect.

Removed because something replaced them: the stand-alone
`builders/heatmap_builder.py` (now the heatmap builder),
`misc/fetch_tissue_tile.py` (now `builders/slides/`) and
`providers/model_provider.py` (now the provider contract).

## How traffic flows

Immediate raw inference, `POST /v1/models/{m}/run`:

```text
client ─HTTP─▶ API ─HTTP─▶ Ray proxy ─▶ model replica      (and back)
```

The API hop adds auth, capacity limits and tile validation that the baseline's
direct proxy access lacked. Scheduled raw inference stores the payload on a
shared volume so it survives an API restart.

Builder run (immediate or scheduled):

```text
client ─HTTP─▶ API ──(scheduled: PostgreSQL row, worker claims it)
                │ HTTP, JSON control data only
                ▼
         builder replica ─Ray handle per tile─▶ model replica (@serve.batch)
                │ reads the slide and writes the output on shared storage
                ▼
         path of the output ─▶ API ─▶ client / EMPAIA publication
```

Slides, tiles, heatmaps and embeddings never cross HTTP on this path: the
builder reads the slide from shared storage, sends tiles to the model as Ray
objects inside the cluster, and writes the result next to the slide. Only
the request and the output's path travel over HTTP. The API reaches builders
over HTTP rather than a Ray handle because the scheduler runs outside Ray
Serve ([ADR 0001](adr/0001-run-scheduler-outside-ray-serve.md)).

Measured on the EMPAIA test instance, a prostate slide with 2,553 tissue tiles
scores in 34–35 s (23 ms read and 70 ms model per tile, tiles in flight
concurrently); the whole EMPAIA job takes 49 s warm and 101 s with a cold GPU
([record](operations/empaia-score-grid-2026-10-31.md)).

## Open

- Tiles go to the model one call each; replica-side `@serve.batch` groups
  them, but client-side batches would cut per-call overhead further.
- The raw inference proxy buffers the model's response before returning it.
- A standalone EMPAIA result is not yet shown by the case browser.
