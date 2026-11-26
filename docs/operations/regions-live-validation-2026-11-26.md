# Regions and viewer v2 on rationai-test, 2026-11-26

Live checks of revision 22 (images `unit-ce40c96`) on the Helm release
`model-service` in `rationai-infra-test-ns`. What changed since revision 19
(`unit-51a62a0`): regions of interest are a GeoJSON `regions` file instead
of a mask TIFF, EMPAIA polygon, circle and collection converters, the
`region_fraction` sink, and the case browser's viewer config in syntax
version 2. Local results are separate: `uv run pytest` passed (523 passed,
59 PostgreSQL tests skipped without `TEST_POSTGRES_DSN`).

## Deployment

- Images: code-refresh layers over the `unit-51a62a0` images. The Ray images
  already carry shapely and rasterio through ratiopath.
- The rendered manifest differed from the live one in the catalog ConfigMap
  (the prostate app's viewer `slide` entry removed) and the four image tags.
- Health, deployments and catalog: loaded, ready, no missing routes. The
  control plane registered app `d3560de0` (standalone, preprocessing) for
  `prostate-heatmap` and PUT its global configuration without error.
- `tumor-purity` is not deployed here: it needs EPISeg, whose weights this
  cluster can't reach.

## Results

- **Golden references**, `pytest -m live tests/live` for
  `prostate-classifier-1` / `prostate-heatmap` (raw route and a pipeline run
  on both golden slides): 3 passed in 4 min 56 s, GPU worker start included.
  A MIG slice was free this time. The digest (`4d68c49caf25`) is unchanged,
  so the TensorRT engine was reused.
- **Regions through the public API.** `POST /v1/pipelines/prostate-heatmap/run`
  on slide `67df0411` with a `regions` file of two regions (a 3072 px square
  and a triangle, level-0 pixels) took 4.9 s. Compared with the golden
  heatmap: all 57 cells that overlap a region were scored, within 1.2e-4 of
  the reference (live tolerance 1e-3). No cell more than one cell away from
  a region was scored; 98 cells in all, against 2427 for the whole slide.
- **EMPAIA preprocessing**, `tools/request_preprocessing.py` on slide
  `7a072fd9`: job `955a6494`, execution `4306529a`. The run succeeded; the
  publication failed, and JES ended the job with exit code 1 ("EMPAIA
  pixelmap upload failed"). Every tile PUT got 500 from the App Service,
  which passed on a 500 from the instance's `test-mds-sdb-annotation`
  service, which logged `Exception in send_and_recv_multi Connection Timeout`.
  That service was mid-rollout: its new replica set (149 min old) had
  restarted 33 times and was not ready. The other app the same preprocessing
  request started (`bc636862`, not model-service's) failed as well. This is
  the EMPAIA test instance, not model-service; rerun once it is healthy.

## Not validated live

- The viewer v2 config in Jirka's case browser, and a standalone job with
  drawn polygons: both need a person in the case browser, and the
  publication path above was down.
- `region_fraction` and the `tumor-purity` app: EPISeg's weights are out of
  reach on this cluster.
