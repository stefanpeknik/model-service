# Full-WSI deployment for prostate heatmap

This preset replaces the dummy workflow in the `model-service-validation`
release in `peknik-ns`. It does not submit an execution. The full-WSI test belongs
to the separate testing session described in
`/tmp/model-service-full-wsi-handoff-2026-10-02.md`.

## Preparation status, 2026-10-02

The scheduler and CPU head images are built locally with source revision
`f1c4f89`. Their packaged source matches the local tree. The source checksums,
including relative filenames and file contents, are:

- Scheduler, 44 Python and workflow-definition files:
  `872698bce8e74aeb0f1db789c5dab27f2194b62795f47fed2e73e52e1ba20294`.
- Head, 32 Python and workflow-definition files:
  `d8d896cd814971ba630150461d4d6433e49b0efc1b91f86ae82b53fd7ba70c65`.

The head imports both native applications and validates the exact rendered
Serve configuration with Ray 2.58. Helm lint and Kubernetes server-side dry-run
passed. The local suite passed 143 tests with 15 skipped.

Deployment is **not ready**. Podman has no CERIT login, and the GPU base-image
manifest read returned unauthorized. The GPU image has not been built or
published; neither locally built image has been published. No Helm upgrade or
namespace mutation has been performed. The live workflow still selects the
dummy implementation. Restore registry access with `podman login cerit.io`,
verify the GPU base tag, then build, publish, and deploy before the testing
session starts. Do not submit the real WSI against the old deployment.

## Deployment

Build and publish these images in the existing `cerit.io/stefanpeknik` project:

| Image | Build file |
| --- | --- |
| `model-service-scheduler:full-wsi-20261002-1` | `docker/Dockerfile.execution-scheduler` |
| `model-service-ray-validation:full-wsi-20261002-1` | `docker/Dockerfile.workflow-head` |
| `model-service-workflow-gpu:full-wsi-20261002-1` | `docker/Dockerfile.workflow-gpu` |

Pass `--build-arg SOURCE_REVISION=<source commit>` to each build. The GPU image
packages local source, including unchanged `models/`, over the existing CERIT
Ray 2.58 GPU base. Neither application fetches a GitHub working-directory archive.

Deploy only this namespace and release:

```sh
helm upgrade model-service-validation helm/rayservice \
  -n peknik-ns --reset-values -f helm/rayservice/values.full-wsi.yaml
```

Do not use `--reuse-values`: the previous release selected a dummy application
and a 600-second deadline. A successful Helm upgrade is not evidence that the
GPU worker or model is ready. Check RayService application status, pod readiness,
and `/healthz/workflows` before admitting the real test.

The native workflow and real classifier are in the same RayService. One warm
MIG 2g.20gb worker provides eight CPUs and 24 GiB RAM. The model uses four CPUs
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

The public origin remains
`http://model-service-validation-execution-scheduler.peknik-ns.svc.cluster.local:8080`.
For a local SDK client:

```sh
oc -n peknik-ns port-forward svc/model-service-validation-execution-scheduler 18080:8080
```

Use `http://127.0.0.1:18080` as the SDK `base_url`. Scheduled calls use
`POST /v1/workflows/prostate-heatmap/schedule`, followed by
`GET /v1/executions/{id}`. Cancellation is
`POST /v1/executions/{id}/cancel`. Raw inference continues forwarding to the
existing production RayService in the same namespace; its origin is unchanged.

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
