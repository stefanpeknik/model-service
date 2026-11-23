# What this thesis added to model-service

The starting point is the team's model-service at commit `3d37a6e` ("Fix:
issues (#9)"): Ray Serve model classes for a binary classifier,
semantic segmentation, Virchow2 and Prov-GigaPath, a heatmap script and
Helm charts. Callers reached each model through the Ray HTTP proxy.

What the department can do now that it could not before: queue many slides
to run overnight as GPUs free up, offer models to researchers as EMPAIA apps,
and add a trained model or a whole-slide job as one YAML file, without
writing Python or rebuilding an image.


## Added

| Area | What it gives | Where |
| --- | --- | --- |
| Public API | One HTTP entry point for analyses, executions and the catalog, with auth hooks, capacity checks and health | `src/model_service/api/` (incl. `api/http/`) |
| Durable scheduler | Executions stored in PostgreSQL: queueing until every served model a run calls has capacity, leases, retries, deadlines, cancellation, priorities | `src/model_service/api/scheduler/` |
| Components | Small reusable classes of seven kinds (loader, runtime, step, source, sink, EMPAIA converter and publisher) that catalog files name, found by a folder scan, each with a conformance suite | `src/components/`, `src/model_service/registry/` |
| Catalog | Served-model and analysis files with per-cluster patches, checked before deployment with errors naming the file, the YAML path and a fix; the API reloads it without restarting | `catalog/`, `src/model_service/catalog/` |
| Served models | One generic Ray Serve host for every trained model, checking each batch against the file's declaration and reporting a digest of what it computes | `src/model_service/served_model/` |
| Analyses | One generic runner for every whole-slide job: heatmaps (score grids) and tile embeddings (HDF5) today, in versioned data formats | `src/model_service/analysis_runner/`, `src/model_service/formats/` |
| EMPAIA provider | Self-registration of each exposed analysis as an EMPAIA app, JES job execution in STANDALONE and PREPROCESSING modes, publication of primitives and pixelmaps through the EMPAIA SDK | `providers/empaia/`, [EMPAIA integration](architecture/empaia.md) |
| Newcomer tooling | Three guides, reference docs and JSON Schemas generated from the code, a catalog test | `docs/guides/`, `docs/reference/`, `tools/generate.py` |
| Tests | About 500 unit, catalog, chart and PostgreSQL tests, plus live checks against golden references recorded on the cluster | `tests/` |

## Preserved

The team's model maths is unchanged. The four model classes became
served-model files on the generic host, built from the `onnx` and `timm`
runtimes and a few steps (`resize`, `center_crop`, `normalize`,
`cls_mean_pool`). Each file reproduces its class bit for bit, known quirks
included: Prov-GigaPath still crops its 224 px tile to the centre, and EPISeg
still runs on ONNX Runtime's default session options because its class passed
them under a keyword ONNX Runtime ignores. Each class was compared with its
file on stand-in weights in the tests, and on the cluster against the
[golden references](operations/golden-references-2026-11-22.md), before it
was deleted. EPISeg had only the first check: the test cluster can't reach
its weights. The fixes come later, under new model IDs.

The baseline's heatmap script became the `prostate-heatmap` and
`episeg-heatmap` analyses, and the team's tiling and tissue code is reused
in `slides/` and the `slide_tiles` source.

## How traffic flows

Raw per-tile inference does not go through model-service's API: a caller
posts a tile straight to the served model's Ray Serve route
(`POST /<model-id>/`), as in the baseline.

Analysis run (immediate or scheduled):

```text
client ─HTTP─▶ API ──(scheduled: PostgreSQL row, worker claims it)
                │ HTTP, the resolved analysis and control data only
                ▼
         analysis runner ─Ray handle per tile─▶ served model (batched)
                │ reads the slide and writes the outputs on shared storage
                ▼
         paths of the outputs ─▶ API ─▶ client / EMPAIA publication
```

Slides, tiles, heatmaps and embeddings never cross HTTP on this path: the
runner reads the slide from shared storage, sends tiles to the model as Ray
objects inside the cluster, and writes the result next to the slide. Only
the request and the outputs' paths travel over HTTP. The API reaches the
runner over HTTP rather than a Ray handle because the scheduler runs outside
Ray Serve ([ADR 0001](adr/0001-run-scheduler-outside-ray-serve.md)).

Measured on the EMPAIA test instance before the rebuild, a prostate slide with 2,553 tissue tiles
scores in 34–35 s (23 ms read and 70 ms model per tile, tiles in flight
concurrently); the whole EMPAIA job takes 49 s warm and 101 s with a cold GPU
([record](operations/empaia-score-grid-2026-10-31.md)).
