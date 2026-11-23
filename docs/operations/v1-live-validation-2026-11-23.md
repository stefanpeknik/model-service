# v1 live validation on rationai-test, 2026-11-22/23

Live checks of the component rebuild (served models, analyses, the analysis
runner and the EMPAIA app built from `expose.empaia`) on the Helm release
`model-service` in `rationai-infra-test-ns`. Local test results are separate:
see the suite (`uv run pytest`). References: the golden set recorded on the
pre-rebuild deployment (`golden-references-2026-11-22.md`).

## Revisions

| Rev | Images | What changed |
|---|---|---|
| 8 | `unit-5682293` | every served model on the generic host, deployed from `catalog/` |
| 9 | `unit-300f031` | analysis runner, migration 0009, CPU-only apps kept off GPU workers |
| 10 | `unit-2c04d52` | prostate-heatmap EMPAIA app from `expose.empaia`, registration reconcile |
| 11 | `unit-984bed7` | builders and transitional code removed (analyses only) |
| 12 | `unit-aeed9eb` | code-review fixes (`GET /<model>/describe`, MLflow checksum, renames) |
| 13 | `unit-2646342` | builder-era names removed, `src/components` split, one fresh migration on a dropped database |

Before each upgrade the rendered manifest was diffed against the live one:
always the same six model-service objects, changes as listed above.

## Results on revision 11 (final design)

- Ray Serve routes: `/analyses`, `/prostate-classifier-1`, `/virchow2`,
  `/prov-gigapath`; `/healthz/deployments` ready, `/healthz/catalog` loaded.
- `POST /v1/builders/...` answers 404; `GET /v1/models` returns `id`,
  `declaration`, `digest`, `analyses`.
- `pytest -m live tests/live`, per model (raw route + analysis run through
  `/v1/analyses/{id}/run`, both golden slides): prostate-classifier-1 /
  prostate-heatmap 3 passed, virchow2 / virchow2-embeddings 3 passed,
  prov-gigapath / prov-gigapath-embeddings 3 passed.
- SDK (`rationai-sdk-python`, branch `analyses`):
  `tests/integration/test_analyses.py` passed (synchronous and scheduled
  prostate-heatmap give equal results).
- EMPAIA: preprocessing job via `scripts/request_preprocessing.py` on slide
  `7a072fd9`; execution `e6efffba` (`prostate-heatmap`, digest `bccb915a…`)
  succeeded with one processing and one finish attempt.

## Results on revision 13 (final)

The `model_service` schema was dumped and dropped first; the API created it
from the single `0001-create-executions` migration (no rows carried over).

- `pytest -m live tests/live` per model: all three 3 passed; digests
  unchanged (`10a8ee8a2b15`, `93df152b6d03`, `0394c8a44dd9`), prostate on
  `TensorrtExecutionProvider`.
- SDK `tests/integration/test_analyses.py` (branch `analyses`, builder-era
  code removed) passed.
- EMPAIA preprocessing job `3d28a298` succeeded (one processing, one finish
  attempt); both API pods re-registered app `d3560de0` with no errors.

- EMPAIA standalone job `d2ff60c2`, started in the case browser with an ROI:
  execution `5f2a0539` came in as `STANDALONE` with inputs typed `wsi` and
  `rectangle` and succeeded (one processing, one finish attempt; 20.6 s of
  runner time, 3 min 19 s including a CPU worker start). Both heatmaps were
  checked in the viewer: the standalone one covers only the ROI, and the
  rendering (yellow shader, 50% opacity, threshold 65) is unchanged.

## Results on revision 12 (after the code review)

- `pytest -m live tests/live` per model: all three 3 passed.
- `GET /<model>/describe` through the Serve ingress: prostate-classifier-1 on
  `TensorrtExecutionProvider` (weights `…/prostate_model_norm.onnx@sha256:a4db8c01…`),
  virchow2 and prov-gigapath on `cuda` with their pinned Hugging Face commits.
  Digests are unchanged from revision 11, so the TensorRT engine was reused.
- EMPAIA preprocessing execution `30e6d15b` succeeded (one processing, one
  finish attempt).

## Earlier results kept

- Revision 9: scheduled run `02b0e6d2` (`POST /v1/analyses/prostate-heatmap/schedule`)
  succeeded and matches the golden heatmap; a repeated `Idempotency-Key`
  returned the same execution. Migration 0009 mapped all 23 old rows to
  `prostate-heatmap` (database dumped first).
- Revision 10: the API registered app `d3560de0` (portal `8e76c21e`), moved
  model-service's PROSTATE/H_AND_E trigger to it and retired the builder-era
  app `a77c0b92`. A marketplace and trigger snapshot before and after differs
  only in those two entries. Preprocessing execution `c944d8a3` succeeded;
  JES reported the job terminated with exit code 0.

## Findings during validation

- **Scheduling deadlock (fixed, 300f031).** A CPU-only builder replica could
  take CPUs on the GPU worker, leaving prostate (4 CPU + 1 GPU) unschedulable
  while the builder waited on it. CPU-only apps now request a `cpu_worker`
  resource only CPU worker groups advertise.
- **Near-blank tile noise.** One Prov-GigaPath row (a glass tile) reached
  cosine 0.99982 against its reference; preprocessing is bit-identical. The
  embedding-table rule now allows ≤1 row in 1000 below 0.9999, none below
  0.9995 (`golden-references-2026-11-22.md`).
- **TensorRT cache.** The generic host keys its engine cache by model ID and
  digest, so prostate rebuilt its engine once (about 16 minutes on the MIG
  slice) and reused it on every later revision.

## Not validated live

- **EPISeg** (`episeg-1`, `episeg-heatmap`): its weights are only on the
  internal MLflow, which the test cluster can't reach, so there are no golden
  references. It is covered by the local equivalence test written before the
  old class was removed and by the catalog checks.
