# EMPAIA SDK Client and score-grid pixelmaps: peknik-ns run, 2026-10-28

model-service now talks to EMPAIA through the SDK's `rationai.empaia.Client`
from the supervisor's MR !10, with a small set of additions on the SDK branch
`feat/job-configuration-client-on-mr10` (raw job data, timeout, write guard,
public direct-access info, `Progress` loaded lazily). The heatmap builder keeps
its score grid, and the publisher builds the pixelmap pyramid from it with the
SDK. Images `empaia-19` and `empaia-20` ran in `peknik-ns` against the EMPAIA
stand-ins (`mock-appservice-7`); the apps were the ones the facade registered
at startup. Nothing ran outside `peknik-ns`; the EMPAIA test instance was only
read.

## Results

Standalone constant-heatmap app (`0d9d12a8-2764-4c9c-89a0-f827bbe9932f`), full
prostate MIRAX slide, the mock's region of interest, `slideAccess: http`:

| Image | Publication | Pixelmap tiles | JES |
| --- | --- | --- | --- |
| `empaia-19` | TIFF path: the grid stayed under the builder's temporary name | 275, level 4 only | `TERMINATED` 0 after 250 s |
| `empaia-20` | Score grid through `ScalarImageOutput.to_pixelmap_pyramid` | 16: level 4: 7, level 5: 5, level 6: 2, levels 7 and 8: 1 each | `TERMINATED` 0 after 182 s |

Both read 1,578 regions through the SDK's `get_region` and finalized.

## Read on the EMPAIA test instance (read-only)

- `GET /v3/{job}/inputs/{slide}` returns the WSI service's slide info with the
  clinical slide's `tissue` and `stain` added, which matches the SDK's strict
  `SlideInfo`.
- The direct-access route is `/v3/rationai/{job}/slides/{input}/direct-access`
  (App Service, then MDS, then CDS). CDS answers `physical_path` under its data
  directory (`/data`) and `storage_path` relative to it; its slide volume is
  mounted at `/data/Projects`. model-service links `physical_path`, so it must
  mount the same volume at the same path.
- The App Service documents `wsi_raw_file_access` in the app's EAD as required
  for that route; neither it nor MDS checks it. Generated EADs declare it.

## Found on the way

- The scheduler image lacked what the SDK imports: numpy, Pillow, h5py,
  imageio. The build adds them; importing `rationai.empaia.utils` also pulled in
  Ray until `Progress` became lazy.
- The SDK wheel (`deploy/images/wheels/`) is baked into both images by the
  code-refresh build until a release carries the additions.
- The mock App Service now answers with UUIDs and the full objects the SDK's
  models require.
