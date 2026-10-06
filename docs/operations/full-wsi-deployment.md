# Full-WSI deployment for prostate heatmap

This preset replaces the dummy workflow in the `model-service-validation`
release in `peknik-ns`. It does not submit an execution. Start at
[the documentation map](../README.md) for current progress. The old
`/tmp/model-service-full-wsi-handoff-2026-10-02.md` is preparation history; the
durable test record is the postmortem linked below.

The October 2 image and deployment records below describe the original warm
checkpoint. The [October 6 validation record](final-standalone-validation-2026-10-06.md)
tracks the later deployment and fault tests. PostgreSQL persistence ownership
and migration safeguards are documented below.

## Phase completion, 2026-10-04

The core end-to-end implementation phase is complete. Scheduled and synchronous
full-WSI invocations through the public RationAI SDK produced identical artifacts,
and the public model-inference path passed its reference check. The
[validation postmortem](full-wsi-validation-postmortem-2026-10-04.md) records the
durable execution state, tile coverage, sampled numerical checks, and resource
observations.

This closes the standalone execution checkpoint, not production or medical
validation. EMPAIA integration, cold startup, fault recovery, and concurrent
workloads remain separate validation work. Calendar or future-dated scheduling
was not part of this implementation contract. Heatmap registration was corrected
locally on October 5; deployment validation and the ONNX Runtime thread-option
mismatch remain follow-ups.

Local persistence-policy and heatmap-placement corrections are recorded in
the [2026-10-05 regression report](../reviews/checkpoint-fixes-2026-10-05.md).
They have not been deployed; image identities and live validation below refer to
the original checkpoint. The corrected unaligned heatmap will have different
pixels and must not be compared against the original artifact hash as a golden result.

## Preparation status, 2026-10-02

The scheduler image uses source revision `f9d52d4`; the CPU head uses `f1c4f89`
and GPU worker uses `0930dfb`. Their packaged application source matched the
October 2 preparation tree, not later local changes. Each checksum hashes sorted
Python/YAML files as relative path,
NUL byte, then file contents:

- Scheduler, 44 Python and workflow-definition files:
  `adeab1bb944ae96944e20c21b970c976f4f83cbcdc1400da51ab3152131d06d4`.
- Head and GPU worker, 32 Python and workflow-definition files:
  `28325948851efbf1a715d7061781dc66b2a6b83d3dded470781e4bee2706a1bf`.

The head imports both native applications and validates the exact rendered
Serve configuration with Ray 2.58. Helm lint and Kubernetes server-side dry-run
passed. The local suite passed 145 tests with 15 skipped. An isolated Mypy check
passed for the three Python modules changed by the final routing fix.

Preparation is complete for the separate session's first real full-WSI test.
All three images are published and verified. `model-service-validation` is
deployed at Helm revision 5, with active Ray cluster
`model-service-validation-jqq6z`. Both `prostate-heatmap` and
`prostate-classifier-1` are `RUNNING`, with `HEALTHY` deployments and no pod
restarts. The dummy application and temporary mock AppService are no longer
deployed. The existing production and storage releases remain at revisions 3
and 1 respectively; neither was changed.

Readiness checks passed:

- Public `/healthz`, `/readyz`, and `/healthz/workflows` return HTTP 200.
- Serve exposes `/workflows/prostate-heatmap` and `/prostate-classifier-1`.
- Deployed scheduler and GPU source fingerprints match those above.
- CUDA sees one A100 80GB PCIe MIG 2g.20gb device, with 19.5 GiB usable memory.
  TensorRT initialization produced a 49,322,060-byte FP16 engine in the new cache.
- The WSI opens at 89,262 by 204,744 pixels, and target MPP selects level 1.
  The staged mask checksum matches the request below.
- Both new 32 GiB PVCs are bound. The worker can write output, attempt-output,
  and scratch mounts as UID 1000; the scheduler can read the inputs and outputs.

Offline image imports, CUDA/TensorRT shared-library resolution, and a small
memmap-to-pyramidal-TIFF check also passed. A temporary restricted pod verified
cluster image-pull access, then was removed. No registry credential was copied
to Kubernetes and no new registry Secret was needed.

No raw-inference or workflow execution request was submitted during preparation.
Full-slide completion, output correctness, inference results, and peak resource
usage remain for the testing session. Before the upgrade, all 168 existing
execution records were terminal with no unfinished provider work.

One non-fatal issue remains in unchanged `models/binary_classifier.py`:
ONNX Runtime reports CPU-affinity errors during startup. The classifier passes
`SessionOptions` as `session_options`, but the installed constructor expects
`sess_options` and ignores the former keyword. Its intended thread settings
therefore are not applied. The replica still initializes and builds its
TensorRT engine. Model source was left unchanged as requested; the testing
session should account for this when checking CPU usage and throughput.

Published image manifests:

| Image tag | Manifest SHA-256 |
| --- | --- |
| Scheduler `full-wsi-20261002-2` | `42aba937efa83059b621218e49275b22ab0ba8611d4b6a1b1f9520e25e549d54` |
| Head `full-wsi-20261002-1` | `704fd69f326d0269e0e571492f2cb8c3e581f2b778158de6e452e3fbac9ad4f5` |
| GPU worker `full-wsi-20261002-1` | `1e1d198b5a1ccc4b01bb7ad127e4f3af1aee9ae9bfd3ab73a880628747706308` |

The local, registry, and deployed GPU image config digest is
`770c36184ba18be64bc23cdc598c59ea8c7044aff4412582af376c92e9626feb`.

## Deployment

The image tags, digests, commands, and first-request payload below record the
October 2 checkpoint. For an authorized deployment of later source, use new
image tags and source revisions, update the Helm preset, and use a fresh
idempotency key. Reusing the recorded key retrieves its old execution, not a new
validation run. Reusing old tags also obscures which geometry produced an artifact.
Verify new artifacts against the corrected geometry, not the historical TIFF hash.

The first real scheduled and synchronous full-WSI runs passed on 2026-10-04.
Results, numerical checks, observed resource usage, and remaining registration
limitations are in the [validation postmortem](full-wsi-validation-postmortem-2026-10-04.md).

Build and publish these images in the existing `cerit.io/stefanpeknik` project:

| Image | Build file |
| --- | --- |
| `model-service-scheduler:full-wsi-20261002-2` | `docker/Dockerfile.execution-scheduler` |
| `model-service-ray-validation:full-wsi-20261002-1` | `docker/Dockerfile.workflow-head` |
| `model-service-workflow-gpu:full-wsi-20261002-1` | `docker/Dockerfile.workflow-gpu` |

Pass `--build-arg SOURCE_REVISION=<source commit>` to each build. The GPU image
packages local source, including unchanged `models/`, over the public
`rayproject/ray:2.58.0-py312-gpu` base. Neither application fetches a GitHub
working-directory archive.

The inference dependencies are pinned to ONNX Runtime GPU `1.23.2`, TensorRT
CUDA 12 `10.9.0.34`, PyTorch `2.8.0`, and torchvision `0.23.0`. The CUDA 12.8
libraries installed with PyTorch take priority over the base image's CUDA 12.1
system libraries. These pins follow the
[ONNX Runtime compatibility documentation](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)
and [PyTorch release instructions](https://pytorch.org/get-started/previous-versions/).
TensorRT libraries come from NVIDIA's hash-verified binary wheel through `uv`;
its PyPI source package is a download stub. OpenSlide is `4.0.1`, distributed in
`openslide-bin==4.0.1.2`, and Ratiopath is pinned to the repository's locked
`1.4.0` release. The GPU build checks the real heatmap and OpenSlide imports;
the older `1.0.3` release does not provide the required mask-builder module.

If the large NVIDIA transfer fails, retain a resumable copy in the ignored
`.cache/container-artifacts` directory. Verify the complete wheel against the
SHA-256 in `docker/Dockerfile.workflow-gpu` before building. Mount that directory
with `--volume "$PWD/.cache/container-artifacts:/opt/build-wheels:ro,z"` and pass
`--build-arg TENSORRT_WHEEL_URL=file:///opt/build-wheels/tensorrt_cu12_libs-10.9.0.34-py2.py3-none-manylinux_2_28_x86_64.whl`.
The wheel mount is build-only; the wheel archive is not copied into the image.
The `runtime-dependencies` build target can complete independently, and its
layer remains cached when the TensorRT transfer needs a retry.
For a retryable dependency download, pass `--build-arg UV_NO_CACHE=false` and
mount the task-owned cache with
`--volume "$PWD/.cache/container-artifacts/uv:/home/ray/.cache/uv:rw,z"`.
Dependency installation runs as root during the build so rootless Podman can
write the host-owned cache; the final image runs as UID 1000. The default build
disables uv caching, and the bind-mounted cache stays outside the image.
The HTTP timeout is 300 seconds instead of uv's 30-second default. Downloads
are limited to four concurrent transfers.

The initial GPU `podman push` failed with Harbor HTTP 502 on large layers.
The upload was switched to a compressed OCI export and `regctl` with 64 MiB
chunks, three concurrent requests, and at most three requests per second.
This keeps individual upload requests short and reuses accepted layers. The
image config digest and uncompressed layer contents stay unchanged.

```sh
skopeo copy --dest-compress-format gzip \
  --dest-oci-accept-uncompressed-layers=false \
  containers-storage:cerit.io/stefanpeknik/model-service-workflow-gpu:full-wsi-20261002-1 \
  oci:.cache/full-wsi-gpu-oci:full-wsi-20261002-1

export REGCTL_CONFIG="$PWD/.cache/upload-tools/regctl.json"
regctl registry set cerit.io --blob-chunk 67108864 --blob-max 67108864 \
  --req-concurrent 3 --req-per-sec 3 --repo-auth
regctl image copy ocidir://.cache/full-wsi-gpu-oci:full-wsi-20261002-1 \
  cerit.io/stefanpeknik/model-service-workflow-gpu:full-wsi-20261002-1
```

Use an existing local Harbor login. Do not package registry credentials in the
image or copy them to Kubernetes. The temporary local uploader configuration
must stay outside the build context. The
[regctl upload options](https://regclient.org/cli/regctl/registry/set/) describe
chunk sizes and request limits. No cluster-side builder or uploader was created.

Deploy only this namespace and release:

```sh
helm upgrade model-service-validation helm/rayservice \
  -n peknik-ns --reset-values -f helm/rayservice/values.full-wsi.yaml
```

The first upgrade hit a Helm 4 server-side apply conflict because the scheduler
image field was owned by `kubectl-set`. Retrying with `--force-conflicts`
transferred that field to Helm without replacing resources. The subsequent
CPU-request adjustment created a replacement pending Ray cluster; its old head
briefly held quota during termination. Both transitional failures are resolved.

Later image updates need a controlled replacement if the warm GPU worker is
still running. Two 23 GiB worker requests alone exceed the namespace's 40 GiB
requested-memory quota, so a zero-downtime RayService upgrade cannot fit.
Quiesce admitted work before replacing this RayService, and preserve PostgreSQL
and the input/output/cache PVCs. Do not change namespace quota or other releases
to work around this limit.

Do not use `--reuse-values`: the previous release selected a dummy application
and a 600-second deadline. A successful Helm upgrade is not evidence that the
GPU worker or model is ready. Check RayService application status, pod readiness,
and `/healthz/workflows` before admitting the real test.

The native workflow and real classifier are in the same RayService. One warm
MIG 2g.20gb worker requests four CPUs and 23 GiB RAM, with limits of eight CPUs
and 24 GiB RAM. CPU throughput is best-effort up to that limit; it does not
reserve eight physical cores on a busy node.
The model uses four CPUs
and the GPU; the workflow uses the remaining four CPUs. Ray's object store is
limited to 1 GiB so both actors' memory reservations fit.

Classifier initialization reads the approved MLflow artifact once on a normal
startup. The tracking URI and artifact are unchanged from the existing deployed
classifier: `https://mlflow.rationai.cloud.e-infra.cz` and run
`169/17dda4dc3f7449478208e87016ce7d86`, checkpoint
`epoch=0-step=6972/prostate_model_norm.onnx`. Metadata is 512-pixel input tiles,
one scalar output, one channel, and target MPP `0.468`. A separate TensorRT cache
directory prevents overwriting the older raw-inference deployment's engine.

EMPAIA admission and its temporary mock AppService are disabled in this preset.
The old mock deployment is removed by the release upgrade; no real AppService is
contacted. EMPAIA remains a follow-up.

## Storage and public caller

The staged input PVC remains unchanged. Helm creates retained 32 GiB output and
32 GiB scratch PVCs. The output PVC is mounted on both scheduler and workflow
worker at `/mnt/validation/outputs` and `/mnt/validation/.workflow-attempts`.
Scheduled output paths are attempt-scoped; the latter mount is essential.
`TMPDIR=/mnt/scratch` also keeps processor and libvips temporary files off the
small input PVC. These sizes are a preparation budget, not measured peak usage;
the test session should monitor them.

### External PostgreSQL persistence, 2026-10-06

The validation PostgreSQL Deployment and Service are both named
`model-service-validation-postgres`. They and the data PVC are managed
separately from the RayService Helm release. The chart consumes the existing
database Secret; it does not provision PostgreSQL or its storage. Preserve these
external resources during Helm upgrades and RayService replacement.

The database Deployment's `database` volume must reference
`model-service-validation-postgres-data` in `peknik-ns`. The
[PVC manifest](../../deployments/peknik-ns/validation-postgres-data.yaml) declares
2 GiB on `nfs-csi` with `ReadWriteOnce`. It creates only the claim and contains no
database credentials or PostgreSQL workload. Existing claims must be inspected
and retained. For an authorized fresh installation where this exact claim is
absent, provision it separately:

```sh
oc -n peknik-ns create -f deployments/peknik-ns/validation-postgres-data.yaml
```

The PostgreSQL Deployment uses one replica and `Recreate` strategy. Preserve its
`postgres:16-alpine` container, UID/GID/fsGroup 70, mount at
`/var/lib/postgresql/data`, existing environment and `PGDATA`, including any
existing data subdirectory. `ReadWriteOnce` alone does
not prevent two database processes on the same node from writing the volume.
Stop the temporary restore pod before starting the Deployment on the claim.

The October 6 migration discovered that the previous database volume was
`emptyDir`. A dump and restore preserved all 176 executions at the migration
boundary, checked with a full-row fingerprint. The subsequent PostgreSQL pod
replacement preserved the same count and fingerprint on the PVC. One queued
execution expired during the maintenance window; that is deadline behavior,
not evidence that it resumed successfully. Detailed outcomes belong in the
October 6 validation record linked above.

For a future migration from `emptyDir`, keep the source pod alive until the
restored database has passed verification:

1. Capture the Deployment and source pod metadata/specification privately with
   mode `0600`, and record their UIDs and the Deployment template fingerprint.
   Quiesce every scheduler process before taking a complete PostgreSQL dump.
   Wait for terminating scheduler pods to leave, and protect the dump with
   mode `0600`. Keep environment values, credentials, and database rows out of
   command output and documentation.
2. Create the new claim and a uniquely named temporary PostgreSQL pod in
   `peknik-ns`. Copy the existing PostgreSQL environment, security settings and
   `PGDATA`; give the pod distinct labels so the existing Service does not route
   callers to it. Restore the complete dump while the original database pod
   remains available for verification.
3. Compare the source and restored `model_service.executions` table counts and
   ordered full-row SHA-256 fingerprints. Print only those safe aggregates.
   Record approval against the source Deployment/pod UIDs, template fingerprint,
   PVC UID/specification, temporary pod UID and recovery dump checksum. Recheck
   these identities and both table fingerprints before stopping the temporary
   pod gracefully. Wait until it is fully gone.
4. With the scheduler still quiesced, check that no other pod mounts the target
   claim and that the source fingerprint has not changed. Patch only the
   Deployment's `database` volume and `Recreate` strategy, preserving its UID and
   all other fields. Test the current UID and resource version atomically so a
   concurrent Deployment change stops the cutover.
5. Wait for PostgreSQL readiness and compare the same table count and fingerprint
   before resuming the scheduler. Verify the database volume references the
   intended PVC. A later PostgreSQL pod replacement must retain the execution
   records; public API readiness alone does not establish persistence.

Before cutover, an aborted migration leaves the original pod authoritative.
After cutover replaces that pod, its `emptyDir` data is gone. Never roll back to
the old `emptyDir` specification as a data recovery step. Keep the scheduler
quiesced and recover from the verified PVC or private dump. Retain both the PVC
and private recovery backup after success. Do not delete the claim, source
database pod, or backup as routine validation cleanup.

The public origin remains
`http://model-service-validation-execution-scheduler.peknik-ns.svc.cluster.local:8080`.
For a local SDK client:

```sh
oc -n peknik-ns port-forward svc/model-service-validation-execution-scheduler 18080:8080
```

Use `http://127.0.0.1:18080` as the SDK `base_url`. Scheduled calls use
`POST /v1/workflows/prostate-heatmap/schedule`, followed by
`GET /v1/executions/{id}`. Cancellation is
`POST /v1/executions/{id}/cancel`. Raw `prostate-classifier-1` calls use the same
warm classifier as the workflow, with the unchanged MLflow artifact. The
`INFERENCE_MODEL_ORIGINS` map selects an exact first path component and preserves
the full path, query, body, and inference headers. Other models retain the
existing production RayService as their fallback origin. The namespace's
40 GiB requested-memory quota cannot accommodate both old and new GPU workers
at once. Sharing the classifier avoids that conflict without changing the old
RayService or the quota; the 23 GiB request leaves room for its CPU worker.
Routing is preserved for other legacy models, but their GPU worker cannot start
alongside the warm full-WSI worker under this quota. Do not treat the fallback
origin as a promise of concurrent GPU capacity for those models.

## Request for the testing session

Submit exactly one scheduled request with a fresh idempotency key. Do not run
the old dummy-workflow demo or the deployed validation suites against this
preset: their small fake inputs and synthetic-output expectations no longer
match the real workflow.

```json
{
  "provider": "standalone",
  "idempotency_key": "full-wsi-20261002-first",
  "requested_timeout_seconds": 21600,
  "body": {
    "inputs": {
      "slide": {
        "kind": "shared_path",
        "path": "wsi/962ea217-0fd7-4ddd-a1a0-165c89056cef/67df0411-395d-5d73-9c2f-3e0ffef45491.mrxs"
      },
      "region_of_interest": {
        "kind": "shared_path",
        "path": "wsi/962ea217-0fd7-4ddd-a1a0-165c89056cef/tissue-mask-open-slide.tiff"
      }
    },
    "resolved_configuration": {
      "output_path": "outputs/full-wsi-prostate-heatmap.tiff",
      "stride_fraction": 0.5,
      "output_tile_height": 256,
      "output_tile_width": 256
    },
    "input_access": {
      "kind": "shared_storage",
      "state": {"root": "/mnt/validation"}
    }
  }
}
```

The default execution deadline is six hours, capped at 24 hours. This includes
queueing, processing, and provider finish. The TIFF path returned in the
successful execution result is authoritative; it includes the processing
attempt namespace and need not equal the requested relative output path.
The prepared mask hash is
`64cb47eec2737bd0441753f7646cead8f169146b90d9c5efc135c7db67dc0806`.
Review its preview and validate final slide alignment in the testing session.

Synchronous execution remains available at
`POST /v1/workflows/prostate-heatmap/run`, but should follow a successful
scheduled full-slide run with suitably long client timeouts. No full-slide or
raw-inference request is needed to prepare this deployment.
