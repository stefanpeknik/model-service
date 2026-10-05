# Final standalone validation, 2026-10-06

Status: in progress. This records preparation and measured outcomes for the
requested final build, deployment, and validation, not a completed sign-off.
Scheduling remains workflow-only, confirmed by the user on October 6. Raw-model
calls retain their existing synchronous contract. EMPAIA remains disabled.

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

Ruff passed for the corrected classifier. GPU startup and public inference
compatibility still need validation in the new deployment.

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

## Validation still required

- Publish and deploy uniquely tagged images containing the corrected source.
- Verify source/image identity, native imports, mounts, and public readiness.
- Revalidate real immediate and scheduled full-WSI output with corrected geometry.
- Exercise replay/conflict, queue persistence, polling, cancellation, deadlines,
  scheduler replacement, and controlled Ray unavailability/recovery.
- Exercise real legacy raw inference sequentially within namespace GPU quota;
  distinguish transport checks from numerical model evidence.
- Validate legacy heatmap compatibility without relying on mutable upstream code
  or routing to a different model artifact.

The old dummy-input deployed suites and old verification scripts are not suitable
unchanged. They assume different fixtures, literal path equality, old pod IDs,
or the pre-correction padded-grid geometry. Keep original checkpoint hashes as
historical evidence rather than expected hashes for this new run.
