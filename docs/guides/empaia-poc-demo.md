# EMPAIA on model-service: proof-of-concept demo

A job started from the EMPAIA test instance's case browser runs a RationAI model
on model-service instead of an app container, and its heatmap goes back to
EMPAIA. Nothing in EMPAIA knows about model-service except one generic JES
setting.

```text
case browser ── slide job ──► Workbench Daemon ──► JES 0.6.15-1
                                                    │ app image http::model-service/builders/heatmap/models/prostate-classifier-1
                                                    ▼
                         model-service (peknik-ns)  POST /v1/empaia/builders/heatmap/models/prostate-classifier-1
                           admission: reads the job's configuration and inputs from App Service
                           builder (Ray, CPU): downloads the slide level through App Service regions
                           model (Ray Serve, GPU MIG slice): prostate-classifier-1
                           publishes the heatmap pixelmap, finalizes the job
                                                    ▲
JES polls the execution (GET Location); cancel and timeout send DELETE
```

## What is deployed

| Where | What |
| --- | --- |
| `rationai-infra-test-ns` | JES `0.6.15-1` with `JES_HTTP_TARGETS={"model-service": "http://model-service-validation-execution-scheduler.peknik-ns.svc.cluster.local:8080/v1/empaia"}` (patched on the Deployment; restore from the deployment repo afterwards) |
| test marketplace | **Prostate Heatmap** (`ProstHM`, preprocessing, slide only), image `http::model-service/builders/heatmap/models/prostate-classifier-1` |
| `peknik-ns` | model-service, Helm release `model-service-validation`, images `empaia-4`, `slideAccess: http`, `dnsSearches: [rationai-infra-test-ns.svc.cluster.local]` |

## Before the demo (30 minutes ahead)

1. A GPU slice must be free: the prostate model runs on a MIG `2g.20gb` slice.
   Warm it with one scheduled tile; a replica that stays unplaced shows
   "waiting for capacity" on the execution:
   ```sh
   oc -n peknik-ns exec -i deploy/model-service-validation-execution-scheduler -- \
       python - prostate-classifier-1 512 < scripts/warm_model.py
   ```
   The replica scales back to zero two minutes after the last request, so warm it
   shortly before starting the job.
2. Readiness: every builder and model route is deployed.
   ```sh
   oc get --raw /api/v1/namespaces/peknik-ns/services/model-service-validation-execution-scheduler:8080/proxy/healthz/deployments
   ```
3. Pick the slide: the whole slide is processed (no quality-control mask yet),
   so the smallest slide in the case keeps the run short.

## During the demo

1. Case browser → Winter School – Prostate → slide → **Zadat úlohu snímku** →
   **Prostate Heatmap**. The Workbench Daemon picks the job up within 30 seconds.
2. Follow it:
   - JES: `oc -n rationai-infra-test-ns logs -f deploy/test-jobs-jes | grep -E "Running Job|Finished Job"`
     shows the stored handle `http::…/v1/empaia/executions/<id>` and the end.
   - model-service: `oc get --raw /api/v1/namespaces/peknik-ns/services/model-service-validation-execution-scheduler:8080/proxy/v1/executions/<id>`
     (`running`, then `succeeded`; a wait for GPU capacity is explained in `message`).
   - App Service traffic: `oc -n rationai-infra-test-ns logs -f deploy/test-jobs-app-api | grep <job-id>`
     (region reads, then the pixelmap upload and `finalize`).
   - Ray dashboard (`oc -n peknik-ns port-forward svc/model-service-validation-head-svc 8265`):
     Serve → `builder-heatmap` and `prostate-classifier-1` replicas.
3. The job ends as succeeded in the case browser; JES records `TERMINATED` 0.

## Points to make

- JES change is generic and upstreamable: an app's `registry_image_url` prefix
  (`http::<target>/<path>`) picks an HTTP execution backend; a documented
  submit/poll/cancel protocol; no EAD or marketplace format change.
- Apps are a builder plus a model: one YAML file per app generates the EAD and
  the registration; adding a model is one model YAML.
- Durable execution: idempotent submission per job, restart-safe handles,
  cancellation and timeouts reach model-service, failures are reported to App
  Service before JES ends the job.
- Slides are read over the App API when storage is not shared
  (`slideAccess: http`), so model-service runs in its own namespace and quota.

## Known limits

- The case browser here cannot display continuous pixelmaps; the heatmap is
  stored but not drawn.
- Whole slides without a quality-control mask spend most of the time on
  background; batching tiles per model call is not done yet.
- No authentication between JES and model-service yet (cluster-internal).
- JES `0.6.15-1` drops the `-memory` resource limits for container apps.
- Registering apps in the test marketplace needs a service-client token
  (LS-AAI), so it is run from inside the JES pod.

## Afterwards

Restore the test namespace from the deployment repo (JES image and env). The
GPU replica scales to zero on its own; the test marketplace entry can stay or be
deleted with `DELETE /v1/custom-mock/portal-apps/<portal-app-id>`.

Validation runs behind this demo: [peknik-ns stand-ins](../operations/empaia-jes-mock-2026-10-21.md),
[slide over the App API](../operations/empaia-slide-over-http-2026-10-23.md),
[first test-instance job](../operations/empaia-test-instance-2026-10-25.md).
