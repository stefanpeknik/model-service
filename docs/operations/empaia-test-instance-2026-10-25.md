# First EMPAIA test-instance job on model-service, 2026-10-25

A job started from the EMPAIA test instance's case browser
(`rationai-infra-test-ns`, testrat.dyn.cloud.e-infra.cz) ran on model-service in
`peknik-ns` and finished: JES recorded `TERMINATED` 0, and the App Service
received a heatmap pixelmap and the finalization. It was the first job on a
real EMPAIA installation rather than stand-ins.

## Setup

| Piece | What ran |
| --- | --- |
| JES | RationAI's `job-execution-service:0.6.15-1` (our HTTP backend plus hardening) in place of `0.6.15-memory`, with `JES_HTTP_TARGETS={"model-service": "http://model-service-validation-execution-scheduler.peknik-ns.svc.cluster.local:8080/v1/empaia"}`; patched on the Deployment, to be restored from the deployment repo afterwards |
| App | ConstHMPre (`deploy/empaia/apps/constant-heatmap-preprocessing.yaml`): heatmap builder with the CPU constant model, preprocessing mode, slide only |
| Registration | Portal app posted to the test marketplace mock (`0.2.28-1`, LS-AAI auth) from inside the JES pod with a client-credentials token for JES's service client; run by the user |
| model-service | `peknik-ns`, images `empaia-4` (`af00346`), `slideAccess: http`, `dnsSearches: [rationai-infra-test-ns.svc.cluster.local]` so the Workbench Daemon's short App Service name resolves |
| Start | "Zadat úlohu snímku" in the case browser, Winter School – Prostate case |

## What happened

| Step | Observed |
| --- | --- |
| Daemon → JES → model-service | `POST /v1/executions`, then `POST /v1/empaia/builders/heatmap/models/constant-classifier` → `202`; JES stored `http::…/v1/empaia/executions/<id>` |
| Admission | Configuration read; `region_of_interest` probed, `404`, left out |
| Slide download | Level 1 of a slide the size of the acceptance prostate slide (44.6k × 102k px), about 17,600 region reads through the real App Service (App Service → MDS → CDS OpenSlide plugin), about 5 minutes; 142 MB tiled TIFF |
| Inference | About 1.12 million constant-model calls (128-px tiles, stride 0.5, whole level, no mask), about 350/s, about 57 minutes |
| Publication | Pixelmap created, 17,600 tiles on slide level 1 (`204` each), `finalize` `200`; about 13 minutes |
| JES | `TERMINATED` 0, about 76 minutes after start |

## Found on the way

- The test instance has no UI for standalone apps; the case browser starts
  preprocessing apps (one slide input) only. Model-service now leaves out optional
  builder inputs the job does not have, and apps can be registered in
  preprocessing mode.
- The marketplace mock switched from `disable_auth` to LS-AAI on 2026-08-29
  (deployment repo `ab23341`); user tokens from the browser have the wrong
  audience, so registration needs a token for the service client.
- `0.6.15-1` drops the `-memory` CPU/memory limits and requests; container apps'
  job pods fall back to the namespace LimitRange (512 Mi, 1 CPU). Raised with the
  image's author.
- Whole-slide preprocessing without a mask spends most of the time on background.
  Background skipping is to come from the quality-control integration; batching
  tiles per model call is to be researched (per-call overhead dominates the 1.5 ms
  model time).
- The builder's own progress logs (e.g. the download line) are not emitted on the
  Ray workers.
