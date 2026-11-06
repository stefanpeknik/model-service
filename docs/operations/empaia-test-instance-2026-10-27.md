# Prostate heatmap on the EMPAIA test instance, 2026-10-27

A Prostate Heatmap preprocessing job (ProstHM,
`deploy/empaia/apps/prostate-heatmap-preprocessing.yaml`) started from the test
instance's case browser ran on model-service `empaia-12` in `peknik-ns` and
finished: JES recorded `TERMINATED` 0 about 7 minutes after model-service accepted
it. The same app took about 19 minutes on 2026-10-25, 11 of them publishing
(see [the 2026-10-25 record](empaia-test-instance-2026-10-25.md)).

## Setup

| Piece | What ran |
| --- | --- |
| JES | The 2026-10-26 testrat update had restored `production/job-execution-service:0.6.15`, an upstream build without the HTTP backend (no `container_clients/http.py` or `routing.py`). The user set the Deployment's image back to `production/job-execution-service:0.6.15-1`; `JES_HTTP_TARGETS` had survived the update. The Deployments repo still pins `0.6.15` |
| model-service | `peknik-ns`, images `empaia-12`: job mode recorded at admission, `Idempotency-Key` on every schedule route, heatmaps published at native score resolution from the builder's TIFF pyramid |
| Start | Case browser, preprocessing job of ProstHM on the prostate slide `7a072fd9` |

## What happened (UTC)

| Time | Step |
| --- | --- |
| 06:45 | `POST /v1/empaia/builders/heatmap/models/prostate-classifier-1` → `202`; admission read the configuration and the job's mode from the real App API |
| 06:45:48 | Slide level 1 download started (44,631 × 102,372 px, 17,600 regions) |
| 06:51:09 | Download done (about 5.5 min); tissue on 2% of the slide |
| 06:51:37 | 69,426 grid positions processed in 28 s, most skipped as background |
| 06:52:13 | Builder returned (385 s in total, TIFF written) |
| by 06:52:49 | Pixelmap published and job finalized; execution `succeeded`, JES `TERMINATED` 0 |

Publication took at most 36 s, against 11 minutes on 2026-10-25. The facade did
not restart.
