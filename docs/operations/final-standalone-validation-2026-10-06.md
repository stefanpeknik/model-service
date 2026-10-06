# Final standalone validation, 2026-10-06

Status: standalone workflow checkpoint validated. Full legacy-model sign-off
remains blocked by the EPISeg artifact and simultaneous-capacity limits below.
Scheduling remains workflow-only, confirmed by the user on October 6. Raw-model
calls retain their existing synchronous contract. EMPAIA remains disabled.

## API call paths

```text
POST /prostate-classifier-1
  execution_scheduler/inference.py: HttpInferenceProxy
    Ray model HTTP ingress -> BinaryClassifier.predict
  <- prediction, same open request; no execution record

POST /v1/workflows/prostate-heatmap/run
  api.py -> synchronous.py -> dispatch.py
    Ray ProstateHeatmapApplication -> WorkflowApplication
      ProstateHeatmapWorkflow -> SlideHeatmapProcessor
        RayHeatmapPredictor -> BinaryClassifier.get_config / predict
  <- named artifact references, same open request; no execution record

POST /v1/workflows/prostate-heatmap/schedule
  api.py -> scheduler.py -> standalone.py -> postgres.py: commit admission
  <- 202 with execution ID
  worker.py: claim + renew lease -> dispatch.py -> same workflow path above
    postgres.py: settle accepted result -> succeeded

GET /v1/executions/{id}        -> stored public execution state
POST /v1/executions/{id}/cancel -> fence durable state, then best-effort local stop
```

Raw inference keeps the existing SDK model methods. Workflow inference uses a
Ray model handle rather than making the public raw-model HTTP call. Only the
scheduled path creates a durable row and retry lifecycle. Scheduling means
queueing for execution when capacity permits, not future-start or recurring
calendar scheduling. Monitoring is per known execution ID; there is no aggregate
queue-list API. `/healthz/workflows` checks route reachability and registration,
not whether a GPU replica is currently ready. `/readyz` checks durable API storage.

## Local baseline

The existing suite ran against an isolated local PostgreSQL 16 container on
loopback, not the deployed database. All 180 tests completed in 7.926 seconds;
175 passed and 5 skipped. Four skips require the native heatmap profile and one
requires libvips. Deployed and local-Ray opt-ins were disabled.

The first sandboxed run could not connect to that loopback database. The complete
rerun with local-network access passed. Those connection failures are not scheduler
test failures. No namespace resources were changed by the local tests.

## Approved classifier correction

The user approved changing `models/binary_classifier.py` from `session_options`
to ONNX Runtime's `sess_options`. A tiny model bundled with ONNX Runtime exercised
the actual classifier `reconfigure` method inside the retained GPU image, with
local source mounted read-only and networking disabled. The external model
provider supplied only that tiny local fixture; no MLflow artifact was fetched.

For configured intra/inter thread counts of `2 / 1`, the original call produced
`0 / 0` through `session.get_session_options()` and failed the assertion. After
the one-line correction, the same check returned `2 / 1` and passed. CPU fallback
was expected because this offline local container had no GPU driver. This verifies
thread-option application, not GPU inference or a throughput improvement.

Ruff passed for the corrected classifier. Final-image GPU startup and public
inference compatibility passed in the live checks below.

## Final image preparation

All three final images use tag `standalone-final-20261006-1` and source revision
`ed9b7acc94760a5572cf7658d5568dc9a216eaa6`. Scheduler and head builds used
their repository Dockerfiles with cached dependencies and networking disabled.
The GPU source refresh used the retained `full-wsi-20261002-1` image, config
`770c36184ba18be64bc23cdc598c59ea8c7044aff4412582af376c92e9626feb`, as its
base. It copied the six packaged source/configuration directories without
changing the installed native or ML dependencies. A fresh dependency build was
not used because the offline dependency layer was unavailable.

Actual GPU dependency versions are Ray 2.58.0, ONNX Runtime GPU 1.23.2, PyTorch
2.8.0, torchvision 0.23.0, TensorRT CUDA 12 10.9.0.34, Ratiopath 1.4.0,
OpenSlide binary 4.0.1.2, and pyvips 3.2.0.

Packaged Python/YAML fingerprints match local source:

- Scheduler: 44 files,
  `fd91c9e49a5b4197964fcfcb8fd8bc78641bee71328aabd1b3b60e932a883b15`.
- Head and GPU: 32 files each,
  `ccc4cf0a120d622477d1e2bd24d50f06019cdbb28d9fdeed3b9f84f505c9eaa6`.

All eight native geometry, processor, and TIFF-publication regression tests
passed offline inside the exact final GPU image in 1.425 seconds. The first run
omitted the test-only `validation/` mount and failed to import that fixture;
the corrected complete run passed. These are local checks, not live EMPAIA
validation. Helm lint and namespace server-side dry-run also passed.

Published scheduler manifest:
`sha256:d68d9e5c152ac50eb4ad682989c5828ddbe33fe02e1c0fd9d76841a1d31fe964`.
Published head manifest:
`sha256:f2a9d4dc969c674033af30d8c15216da81010f8842a6e7a0c714582ddc23c4c1`.
The direct GPU upload was stopped after it made no useful completion progress.
The final upload reused the original registry manifest's 32 dependency layers,
added one 25,708-byte gzip source layer, and updated the source-revision label.
This has the same dependency filesystem and packaged source as the locally tested
refresh, but a different image config identity. Its deployed source fingerprint
matches the value above.

Published GPU manifest:
`sha256:afd35342a5b8727d172443e2948308cc64d3a86d2e054a587908284825ccad0a`.
GPU config:
`sha256:8ac52d27f294f84a5c50dc8b1d3b1d136d39a829368d91b9ce3ccfb6d26e1efc`.
Added source-layer digest:
`sha256:7bce4e1c88517704d1a19b2738ce862aadfdad4151938355572b6a0e1c326d93`.

## Final deployment

Helm release `model-service-validation` was deployed at revision 6 and updated
to revision 7 for the replica-policy correction below, using
Ray cluster `model-service-validation-lc75c`. Both native applications are
`RUNNING / HEALTHY`. Scheduler, head, and worker report the published manifest
identities above and zero restarts. The running GPU source fingerprint matches
local source. PostgreSQL and all retained PVCs were preserved; the separate
legacy RayService was not changed.

The controlled replacement briefly triggered worker quota rejections while old
pods terminated. These resolved without quota changes. During Ray unavailability,
`/readyz` stayed HTTP 200 and `/healthz/workflows` returned HTTP 503 with
`status=unreachable`. It returned HTTP 200 after Ray became reachable again.

The final classifier's public HTTP/SDK test passed in 1.023 seconds. It exercised
the stored all-zero tile score `0.256591796875`, trailing-slash inference, RGB
array/PIL inputs, and synchronous/asynchronous SDK calls. Other model cases were
deliberately deselected in this run. Filtered replacement startup logs contained
no CPU-affinity or error matches; this is not a throughput benchmark.

## Real WSI results and replica-policy correction

The active public HTTP/SDK suite passed two tests in 157.390 seconds, with its
two claim-paused cases deliberately skipped. Execution
`3c47caeb-0f9d-47ec-bcf5-44e541ed3cd7` succeeded. Admission replay, conflicting
key reuse, terminal replay from a fresh SDK client, and synchronous invocation
all passed. Raw terminal HTTP views retained only the five public fields.
The suite reported that filesystem verification was separate; its success alone
was not used as evidence of image content.

Separate worker-side verification found identical scheduled and immediate TIFF
bytes, with SHA-256
`f7920599c51fb6f60483baa76678edcc4f455423f10d2897bcbe1395aafeacfc`
and size 15,474,188 bytes. The image is single-band uint8, 44,631 by 102,372 pixels.
It reconstructs the 44,800 by 102,400 padded span before cropping. All 20 sampled
reconstruction pixels matched exactly. Four background samples were zero, and
182,032,016 output pixels were nonzero.

Verification made 380 real-tile HTTP inference checks and compared eight with
CPU ONNX Runtime using the cached artifact, with maximum CPU/GPU difference
0.0011797845363616943 against a 0.002 tolerance. No additional MLflow read was
needed. Cached model SHA-256 remained
`a4db8c01735c6b4b992e1735e4bf9c1749e6e1b9dffa20b8e020cd3f0cabc3b1`.
This is model/reconstruction consistency evidence, not clinical validation.

Load exposed inherited Ray autoscaling. Although the preset specified
`num_replicas: 1`, the Python deployments' `num_replicas="auto"` left an effective
autoscaling limit of 100. The classifier attempted three replicas on the one-GPU
worker. A live configuration assertion failed with limits of 100; an offline
Ray 2.58 reproduction confirmed that a numeric override alone retains the
autoscaling configuration. Passing `autoscaling_config=None` clears it.

Both full-WSI application YAMLs now explicitly set `autoscaling_config: null`.
Helm revision 7 applied this without replacing the Ray cluster. The live
configuration assertion then passed with `num_replicas=1`, no autoscaling config,
and exactly one actual replica for each application. Readiness and the public
classifier check were repeated after the change. This is a preset-only fix;
general model autoscaling remains unchanged.

Raw native-suite and geometry evidence is retained under the ignored
`downloads/validation/standalone-final-20261006/` directory.

## Live durable API checks

The scheduler was replaced with the final image. Its running pod reports the
published scheduler manifest above. Tests reached its public HTTP contract
through a loopback-only Kubernetes service port-forward; no external ingress
was configured by this validation.

Processing claims were temporarily paused by a session-level advisory lock for
`prostate-heatmap`. Admission, polling, and independent maintenance stayed active.
One execution, `44209c56-e9bf-4e15-8f95-963652cba18b`, was admitted before
scheduler replacement and was still retrievable as `queued` afterward. It was
then cancelled through HTTP. This proves admission persistence across that
replacement, not resumed inference or lease-loss recovery.

The paused native suite passed three tests in 7.971 seconds. Its successful
WSI-processing case was deliberately skipped:

- `8d368d4d-62b3-4d46-b24e-bb00907252e6` became `cancelled`; repeated SDK
  cancellation, polling, and schedule replay retained that terminal result.
- `3754f26a-d447-44a6-adc2-5afcba6ebcc4` expired after a five-second queued
  deadline without dispatch. Replay and cancellation retained
  `failed / execution_deadline_exceeded`.
- Unknown workflow and execution IDs returned the documented safe HTTP errors.

The task-owned advisory-lock session was explicitly terminated after these
checks. PostgreSQL reported zero queued/running executions and zero pending
work kinds. No credential or protected context was returned in those public views.

## Observed deployment before upgrade

Read-only checks in `peknik-ns` found the existing native validation RayService
with warm prostate workflow/classifier applications and no pod restarts.
The separate legacy RayService declares the heatmap builder, classifier, EPISeg,
Prov-GigaPath, and Virchow2. Reported healthy deployment status does not prove
that scaled-to-zero models can initialize or return valid inference.

The deployed legacy classifier already uses the same public MLflow experiment
169 artifact as the native classifier. Its repository default still references
experiment 65, so source defaults must not be mistaken for live artifact identity.
The legacy EPISeg also uses the public tracking origin. Further model-artifact
and cache checks remain before real legacy inference tests.

## Database persistence and scheduler recovery

The checkpoint PostgreSQL volume was `emptyDir`. Its records would have been lost
on database pod replacement, so the deployed storage did not meet the persistence
goal. The scheduler was quiesced, a private full backup was restored to a new
2 GiB `nfs-csi` PVC, and the source and restored databases matched all 176
execution records with ordered full-row SHA-256
`7a85e2fab8e05ff8f07fd5b8d277dbe871a15683eb79d9a6d8743dd60b6e6495`.
After stopping the temporary restore writer, the external database Deployment
was changed to use that claim and `Recreate` strategy. The replacement pod
matched the same count and hash. A second deliberate database pod replacement
also retained that exact fingerprint. The private backup and PVC were retained.
The [runbook](full-wsi-deployment.md) records ownership and recovery safeguards.

Execution `73f0562f-fc16-4862-90ea-51bb9be329f9`, admitted before the backup,
remained retrievable after database replacement. It had expired during the
maintenance window and settled as `failed / execution_deadline_exceeded` when
the scheduler resumed. This proves retained admission and deadline settlement,
not successful resumption of that execution.

An earlier in-flight crash test completed execution
`5517311c-029e-4bcc-91f4-10b517260e98` after forced scheduler pod replacement,
with observable `running → queued → running → succeeded` across lease expiry.
Ray and PostgreSQL stayed running. A five-second queued execution,
`d9ea3d37-e3da-4f06-b1d1-b48c23b0b79f`, expired while the real WSI execution
was running. This proves independent deadline maintenance under actual load.
Recovery is at least once, not exactly once.

After migration, execution `2285eb27-8040-4bd6-a8c7-f39e6168b809` was admitted
while native GPU capacity was deliberately zero. Readiness, admission and polling
remained available. While workers were absent, the exact native Serve controller
was killed with restart allowed. Its actor retained identity and restarted with
a new worker process; both applications reappeared. The same execution succeeded
after native GPU capacity was restored. Its artifact is byte-identical to the
corrected full-WSI baseline above, with the same SHA-256 and size. It used one
processing attempt. This proves recovery of pending work across a scaled-zero
controller restart and capacity restoration, not the behavior of a permanently
disabled controller.

## Legacy validation limits observed so far

The existing EPISeg worker became Ready and initialized its replica constructor,
but model reconfiguration failed. The configured MLflow artifact request
exhausted client retries after repeated HTTP 500 responses. Ray then attempted
another replica. The selected application and worker were stopped to prevent
continued artifact-download retries; no successful segmentation inference is
claimed. A constructor initialization message or stale healthy status does not
prove that the model is loaded. The external artifact issue was not changed.

The initial temporary profile also failed to apply: Ray 2.53 rejected fixed
`num_replicas: 0` for inactive applications. Kubernetes recorded HTTP 400
`ServeDeploySchema` errors, and the effective Serve configuration retained the
original applications. A later worker therefore picked up another pending
EPISeg replica rather than the selected embedding model. GPU capacity was
stopped again. The validation helper now removes inactive applications
temporarily and verifies the effective profile before starting a worker. The
exact Ray schema accepted `applications=[]`, and an effective-profile assertion
changed from failing with all five original applications to passing with none.
No model implementation was changed for this setup issue.

With that correction, the effective profile contained only `prov-gigapath`, and
all three offline flags were confirmed before worker creation. Its five checked
HTTP/SDK calls passed in 1.765 seconds: RGB array/PIL input, sync/async methods,
float32/float16 agreement, binary content type, output-shape header and decoded
length. One older interrupted setup call also reached that actor when it became
available. No new Hugging Face weights were fetched. Runtime versions were
Ray 2.53.0, PyTorch 2.5.1+cu121 and timm 1.0.25. The worker reported CUDA available
on an A100 MIG 2g.20gb, 5,960,110,080 bytes current cgroup memory, and zero OOM
events. This is transport/model consistency evidence, not an independent
embedding-quality reference.

Virchow's selected-only profile and offline flags passed the same pre-worker
gate. Ten public HTTP/SDK calls passed in 16.620 seconds, including RGB array/PIL,
sync/async use, float32/float16 output, pooled/unpooled shapes and the expected
class-token/patch-mean relation. Its real cached GPU actor returned HTTP 200.
The worker reported the same A100 MIG GPU, 4,064,694,272 bytes current cgroup
memory and zero OOM events. This does not establish an independent pathology
embedding-quality reference.

These legacy checks used one temporary GPU profile at a time, with the native
worker absent and the legacy worker CPU request reduced from eight to four cores
while retaining its eight-core limit. The saved worker policies, environment
and legacy Serve configuration were restored afterward. The restoration check
excluded only KubeRay's changing `scaleStrategy.workersToDelete` bookkeeping.
The native head, scheduler, database and GPU worker report zero pod restarts.

The namespace's 40 GiB requested-memory quota cannot accommodate the native
23 GiB GPU worker and the separate legacy 24 GiB GPU worker together with their
heads and database. The final preset keeps the native worker active, while
legacy workers retain their original scaled-zero policy. Classifier inference
is routed to the native worker. GigaPath/Virchow checks above do not prove they
can cold-start alongside this final native allocation. Capacity swapping or a
reviewed resource/deployment change is needed for an all-model demonstration;
no quota or outside-namespace resource was changed.

The unchanged upstream legacy classifier lacks `get_config`, which the old
heatmap builder requires. That builder also rejects scalar classifier outputs.
Its prostate-classifier combination therefore cannot be counted as compatible.
The new native prostate workflow passed the real-slide checks above. A legacy
builder exercise would require a spatial-output model such as EPISeg and is
currently blocked by its artifact fetch.

The local 180-test suite was repeated after migration preparation: 175 passed,
5 skipped in 7.742 seconds, using the isolated local database, not the live one.

## Final restored-state checks

The final active native public suite passed two tests in 133.556 seconds, with
the two claim-paused cases deliberately skipped because processing was enabled.
Execution `9f042684-b78c-4a47-b407-fc7ed64c44e3` succeeded; replay/conflict,
fresh-client recovery, immediate execution and safe unknown-ID errors passed.
Separate worker-side verification found both returned TIFFs byte-identical to
the corrected baseline, with SHA-256
`f7920599c51fb6f60483baa76678edcc4f455423f10d2897bcbe1395aafeacfc`.
The classifier's five HTTP/SDK calls also passed again in 1.031 seconds.
The recovery case's fresh typed-SDK replay and polling passed. An initial
throwaway recovery script passed HTTP/artifact checks but used an unsupported
SDK keyword; correcting the script to use `WorkflowRequest` passed without any
SDK or service change.

Final PostgreSQL aggregates show 178 retained executions: 114 succeeded,
40 failed and 24 cancelled, with zero pending work kinds and no queued/running
records. `/readyz` and `/healthz/workflows` both returned HTTP 200. Both native
applications are healthy with exactly one replica and no autoscaling policy.
Legacy worker policies and Serve configuration match the saved baseline apart
from controller-owned deletion bookkeeping. Temporary model workers and the
restore pod are gone; the database PVC and private recovery backup are retained.
The recovery archive is `.cache/final-postgres-before-persistence.dump`, with
private migration metadata under `.cache/final-postgres-migration/`. These are
local, ignored files, not repository deliverables. Keep them protected and keep
the existing database/encryption Secret intact. The isolated local test database
container was stopped after the regression checks; deployed storage was not
deleted.

## Validation still required

- Real EPISeg inference remains blocked by its configured MLflow artifact's
  HTTP 500 responses. Changing the external service or selecting a replacement
  artifact is outside this namespace-only validation.
- Legacy heatmap-builder validation remains blocked by EPISeg. The unchanged
  upstream builder/classifier combination is incompatible, as described above.
- Simultaneous legacy and native GPU availability is not established and does
  not fit the current namespace quota with the existing worker sizes.
- EMPAIA remains the next phase. Standalone success does not validate service
  authentication, AppService publication or terminal reporting.
- These checks do not establish clinical accuracy, multi-GPU scaling, high
  availability of PostgreSQL, or production load limits.

The old dummy-input deployed suites and old verification scripts are not suitable
unchanged. They assume different fixtures, literal path equality, old pod IDs,
or the pre-correction padded-grid geometry. Keep original checkpoint hashes as
historical evidence rather than expected hashes for this new run.
