# EMPAIA jobs through JES's HTTP backend: peknik-ns postmortem, 2026-10-21

The JES draft and model-service's EMPAIA route were run together for the first
time, in `peknik-ns`, against EMPAIA stand-ins. Every lifecycle path worked
after four bugs were fixed on the way: a job finishing, failing, being stopped,
timing out, surviving a JES restart, and being resubmitted. A real MIRAX slide
went through end to end on CPU. The GPU run did not happen: the cluster had no
free MIG slice.

## Setup

| Piece | What ran |
| --- | --- |
| JES | Our draft (`jes-http`, branch `feat/http-execution-backend`, image `empaia-jes:http-8d01143`), no Docker daemon or Kubernetes namespace, `JES_HTTP_TARGETS={"model-service": "http://model-service-validation-execution-scheduler:8080/v1/empaia"}`, CPU queues 4 standalone and 2 preprocessing |
| JES state | MongoDB 7, `emptyDir` |
| Marketplace | EMPAIA's `marketplace-service-mock:0.2.42`, auth disabled |
| AppService | model-service's mock (`testing/empaia_app_service.py`), serving the synthetic 256-px fixtures or the acceptance prostate slide (MIRAX, 89,262 × 204,744 px, 0.243 µm/px) and its tissue mask |
| model-service | Branch `claude/empaia-http-protocol`, EMPAIA enabled, scratch volume as the shared EMPAIA workspace |
| Workbench Daemon | `.cache/empaia_job.py` in the worktree: mints a job token, `POST /v1/executions`, follows JES, then reads what the mock observed |

Manifest: `deploy/k8s/peknik-ns/empaia-mock.yaml`. Apps were registered with
`scripts/register_empaia_app.py`, so their images read
`http::model-service/builders/heatmap/models/<model>`. Nothing ran outside
`peknik-ns`.

## Results

| # | Scenario | Outcome | JES saw | AppService saw |
| --- | --- | --- | --- | --- |
| 1 | Constant model, synthetic slide | ✅ 14 s | `TERMINATED` 0 | one pixelmap tile (value 0.749 = 0.75 as uint8), finalized |
| 2 | Missing slide | ✅ 8 s | `TERMINATED` 1, "builder failed unexpectedly" | failure report with the same message, before JES ended the job |
| 3 | Stop through JES after 15 s | ✅ 18 s | `STOPPED` | cancellation report |
| 4 | JES timeout 60 s, real slide | ✅ 61 s | `TIMEOUT` | cancellation report; the execution was cancelled by `DELETE` |
| 5 | JES pod deleted 20 s into a real-slide job | ✅ | the new pod resumed from the stored `http::` handle and recorded the final status | (job then failed in the mock, see bug 4) |
| 6 | Same job posted twice to model-service | ✅ | — | `202` then `200`, same `Location`; ran once (one output, one finalize) |
| 7 | Constant model, real MIRAX slide | ✅ 8 min 53 s | `TERMINATED` 0 | 17,600 tiles on the level matching the model, finalized |
| 8 | Prostate model, real slide (GPU) | ❌ not run | stopped after 25 min | cancellation report |

Scenario 8 waited on a GPU worker that could not be scheduled: "0/51 nodes are
available … 40 cannot allocate all claims", i.e. no free MIG 2g.20gb slice
anywhere on the cluster. It was stopped through JES; the pending worker is
covered under "GPU" below.

## Bugs found and fixed

1. **MIRAX slides could not be read through EMPAIA direct access.** Inputs were
   linked as `inputs/slide` with no extension and without the companion data
   directory, so OpenSlide could neither detect nor read a `.mrxs`. Synthetic
   TIFFs had hidden this. `materialize_input` now gets a directory per input
   and returns the file it made; the link keeps the source name and a `.mrxs`
   brings its directory (`b4ec5bc`, with a regression test). Codex's October
   AppService research had listed MIRAX materialization as an unverified risk;
   it is now confirmed and fixed.
2. **JES could not start without a default backend** (fixed before the run,
   `8d01143`): an HTTP-only JES has no Docker daemon or namespace. It now logs
   the default backend's failure and starts.
3. **JES overwrote the stop reason.** After a stop, the HTTP backend's
   "stopped by remote execution service" replaced JES's "stopped by user",
   because JES lets later updates overwrite the message. The backend no longer
   supplies a default (`3d55022`, not redeployed).
4. **The AppService mock was narrower than a real job.** It accepted one tile
   at level 0 only, sent a retired `output_path` setting in one scenario, and
   chose scenarios by job-ID prefix, which JES forbids (job IDs must be UUIDs).
   It now accepts a real slide's tiles and takes scenarios declared per job
   (`51a216d`, `b4ec5bc`, `60d2485`).

Setup problems, fixed in the manifest or script: EMPAIA's images keep their
venv under `/home/appuser` and must run as uid 1001; the marketplace mock wants
version-4 UUIDs and no `organization_id` on the app record.

## Open issues

These are not fixed and need decisions or work before the test namespace:

1. **A job refused at admission is not reported to EMPAIA.** With a
   configuration model-service rejects (HTTP 422), JES recorded `ERROR` with
   "Remote execution service refused the job: HTTP 422", but nobody told
   AppService why. The Workbench would show a failed job with a generic
   message. Model-service should report the refusal to AppService itself
   before answering 422, like any other failure.
2. **Failure messages hide the cause.** A missing slide and an AppService read
   timeout both surfaced as "builder failed unexpectedly". The input-access
   errors need their own codes and messages.
3. **AppService reads give up after about 5 s.** In scenario 4's first attempt
   (a slow mock read), each builder attempt failed after ~5 s and the job
   failed after two attempts, before JES's timeout. A slow but healthy
   AppService would fail jobs the same way. The timeout and retry policy for
   input reads should be set deliberately.
4. **JES logs a token error every few seconds without an identity provider.**
   With empty IdP settings, JES's sender-auth client still tries to fetch a
   token. Harmless, but noisy; an upstream fix would treat empty settings as
   "no auth".
5. **Queues.** Without an app-requirements entry, remote jobs use the default
   standalone CPU queue and hold a worker slot while model-service computes;
   PREPROCESSING jobs without an entry also fall back to the standalone queue
   (upstream behaviour). Real installations should list remote apps as
   `PROXY`.
6. **No GPU acceptance yet.** The prostate model has not run through JES. Rerun
   scenario 8 when a MIG slice is free.
7. **Authentication between JES and model-service** remains open; this run was
   cluster-internal and unauthenticated.

## GPU

The prostate attempt requested one GPU worker, which stayed `Pending` because
the cluster had no free slice; it never held a GPU. After the job was stopped,
Ray Serve kept its prostate replica in `STARTING` and the autoscaler kept the
pending worker for over 15 minutes: Serve does not scale down a replica that
was never scheduled. Removing the prostate app from Serve and re-applying the
RayService's own Serve config (unchanged) cleared it; the worker went away and
all routes were back. KubeRay did not restore the app on its own, because it
only re-applies the config when the RayService spec changes.

Lesson: a model whose GPU replica cannot be scheduled keeps the request open
until someone resets the app. When the cluster is short of GPUs, check `ray
status` for pending demands after cancelling GPU work.

## Notes for next time

- Port-forwards die whenever their pod is replaced, and a new forward fails
  while an old one holds the port. Running the job driver inside the cluster
  would avoid most of the lost time.
- The Ray image had to be rebuilt because the code layout changed since the
  last deploy; the facade's Helm command is now explicit so code-refresh images
  start the current module.
