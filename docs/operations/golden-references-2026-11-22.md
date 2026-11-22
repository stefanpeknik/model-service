# Golden references, 2026-11-22

Recorded on the test deployment before any component-platform code was
deployed, so each migration step can show its outputs are unchanged. Release
`model-service` in rationai-infra-test-ns ran images `empaia-36` (built from
`0352fa6`); the code those models run is numerically the same at `af75e64`.

- Generator: `tests/live/generate_references.py`; comparisons and tolerances:
  `tests/live/golden.py`; hashes and provenance: `tests/live/manifest.json`.
- Files: `/mnt/shared/golden/` on the claim `model-service-full-wsi-outputs`
  (`slides/`, `refs/raw/`, `runs/`), with a local copy in `.cache/golden/`.
- Slides, copied from the instance's slide volume into `golden/slides/`:
  `67df0411-…` (prostate, 333 MB, 0.243 µm/px) and `97b654fd-…` (139 MB,
  0.199 µm/px).

## Recorded

- Raw routes, 32 tiles per model (16 per slide, on tissue, read at the level
  nearest the model's resolution): `prostate-classifier-1` (JSON scores),
  `virchow2` (pooled `(2, 1280)` plus 4 unpooled `(257, 1280)` replies),
  `prov-gigapath` (`(1536,)`).
- Synchronous runs on both slides with default settings:
  `heatmap` × `prostate-classifier-1` (grids 400×175 and 521×254, 2427 and
  1850 scored cells), `tile-embeddings` × `virchow2` and × `prov-gigapath`
  (1687 and 1311 rows).
- Prov-GigaPath isn't part of the test deployment. For this recording,
  revision 7 of the release (chart from `0352fa6`, same images) added a test
  copy with half a GPU and batches of 64; the diff against the live manifest
  was only that Serve application, the model catalog and the API's catalog
  checksum. The API pods restarted with it, which cut one synchronous run;
  it was repeated.
- Virchow2 and Prov-GigaPath load from Hugging Face `main` (the classes don't
  pin a revision); the manifest records the commits `main` pointed at.

## Not recorded

`episeg-1`: its weights are on the internal MLflow
(`http://mlflow.rationai-mlflow:5000`, experiment 10), which the test cluster
can't resolve, and the public MLflow has no such run. EPISeg is checked only
against stand-in weights in the local suite until the test cluster can load
the model.

## Repeatability

`pytest -m live tests/live -k 'raw or manifest'` against the same deployment
right after recording: the manifest matches and all three raw routes answer
within the live tolerances (scores `atol=1e-3`, embeddings cosine ≥ 0.9999),
so GPU run-to-run variation fits them.
