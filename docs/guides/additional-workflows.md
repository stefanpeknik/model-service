# Additional standalone workflows

Implemented on 2026-10-09 following approval to add slide-processing workflows
above the other raw models. This is a new increment beyond the closed standalone
scheduling checkpoint. EMPAIA integration and live GPU validation remain separate.

## Inputs and outputs

Each workflow accepts a `slide` and a `region_of_interest`, both shared-storage
relative paths. The slide must be OpenSlide-readable and have physical-resolution
metadata. The region is a tiled TIFF mask covering the slide's coordinate extent;
its dimensions and pyramid levels may differ from the slide. Nonzero mask pixels
select tiles. A selected tile may include pixels outside the mask. Edge tiles are
padded by the existing slide reader. These workflows do not accept tile arrays,
URLs, polygon JSON, or EMPAIA job credentials.

The mask need not contain physical-resolution tags. When they are absent, its
finest level is used and dimensions determine the mapping to slide coordinates.

| Workflow | Existing raw model | Output |
| --- | --- | --- |
| `episeg-segmentation` | `episeg-1` | `segmentation`, a floating-point pyramidal score TIFF |
| `gigapath-embeddings` | `prov-gigapath` | `embeddings`, an HDF5 file containing one feature vector per selected tile |
| `virchow2-embeddings` | `virchow2` | `embeddings`, an HDF5 file containing two feature vectors per selected tile |

EPISeg uses the existing spatial reconstruction and overlap averaging. Its output
uses the chosen slide level's dimensions and physical resolution. Scores stay
floating point, without a threshold, class-label interpretation, or multiplication
by 255. A completely empty region produces a zero score map. The existing prostate
heatmap remains an 8-bit TIFF.

Embedding workflows currently read 224-pixel tiles at slide level zero. Neither
embedding model declares a target MPP through its current model contract, so no
training-resolution claim or automatic resampling is made. Actual MPP is recorded
in the artifact. Confirm sampling with the model owner before clinical or
cross-slide comparisons. Tile size belongs to deployment/model wiring, not a job
setting. The workflows use the existing raw HTTP routes to preserve the model's
resize, crop, normalization, and pooling logic. They call Ray's internal origin,
not the public scheduler's inference proxy.

## HDF5 schema, version 1

The media type is `application/x-hdf5`.

- `coordinates`: int64 array of shape `[N, 2]`, ordered `x, y`, giving tile origins
  in original slide level-zero pixels. Rows remain in deterministic grid order,
  regardless of model-response completion order.
- `features`: float32 array of shape `[N, D]` for GigaPath or `[N, 2, D]` for Virchow2.
  The feature dimension `D` comes from the actual raw-model response. Nonfinite
  values, malformed shapes, and dimensions changing between tiles are rejected.
- Attributes: `schema_version`, `model_id`, `summary_names`, `coordinate_space`,
  `slide_width`, `slide_height`, `slide_level`, `mpp_x`, `mpp_y`, `tile_size`, `stride`,
  and `tile_count`.

Virchow2 summary order is `global`, then `local_mean`, matching its existing
default raw output. GigaPath's summary name is `tile`. These are model features,
not diagnoses or calibrated disease probabilities. An empty region makes no model
requests and produces empty coordinates and features; its unknown feature
dimension is zero, giving `[0, 0]` or `[0, 2, 0]`.

The writer appends rows to extensible HDF5 datasets. The number of outstanding
tiles is bounded by worker settings. All files close before atomic publication.
Failed or cancelled runs remove temporary artifacts, preserving any previous
completed output. Scheduled outputs use the existing isolated attempt directory.

## Calling a workflow

The existing API and SDK methods apply unchanged. Immediate calls use
`POST /v1/workflows/{workflow_id}/run`; durable admission uses
`POST /v1/workflows/{workflow_id}/schedule`, followed by the existing execution
polling and cancellation routes. Raw inference remains synchronous and unchanged.

Example immediate GigaPath invocation:

```json
{
  "inputs": {
    "slide": {"kind": "shared_path", "path": "inputs/slide.tiff"},
    "region_of_interest": {"kind": "shared_path", "path": "inputs/mask.tiff"}
  },
  "resolved_configuration": {
    "output_path": "outputs/gigapath.h5",
    "stride_fraction": 1.0
  },
  "input_access": {"kind": "shared_storage", "state": {"root": "/mnt/data"}},
  "deadline_at": "2099-01-01T00:00:00Z"
}
```

Replace the example deadline with a real bounded deadline. For scheduling, put
this invocation in `body`, add `provider: "standalone"` and a unique
`idempotency_key`. The scheduler sets its own effective deadline. The public SDK
expects complete `resolved_configuration` too; it does not fetch or fill defaults.
Python callers inside model-service can use `WorkflowCatalog.resolve_configuration`.

EPISeg configuration defaults are `output_path: "segmentation.tiff"`,
`stride_fraction: 0.5`, and TIFF tile height/width of 256. Both embedding workflows
default to `output_path: "embeddings.h5"` and `stride_fraction: 1.0`.
Use distinct output paths for separate immediate runs. Every configuration field
must be present in `resolved_configuration`, as for prostate heatmap.

## Deployment and validation limits

The default Helm application list includes these workflows and their raw models.
The new workflow ingresses run code from the image, not a GitHub `main` archive.
Rebuild and retag the head/scheduler and the CPU/GPU worker images from this source
before enabling them. Updated Dockerfiles include HDF5 output dependencies; the
dedicated workflow GPU image also includes `timm` for the raw embedding models.
Changing YAML alone does not update existing tagged images or mounted caches.

`WORKFLOW_IDS`, supplied as a JSON array, restricts a scheduler deployment's
catalog. Helm exposes it as `executionScheduler.workflowIds`. Unset means all
built-ins; empty or unknown selections fail startup. The existing full-WSI and
CPU-validation presets explicitly retain only `prostate-heatmap`. No existing
deployment or image was changed by this implementation task.

Raw embedding models still require Hugging Face access and their existing model
cache/credentials. EPISeg still requires its existing MLflow artifact and working
ONNX runtime. Shared input/output storage must be visible at the same paths to the
scheduler and computation replicas. Resource sizing and these live model paths
have not been validated for the new workflows. `models/` was left unchanged.

Local regression coverage lives in `tests/test_additional_workflows.py`, through
public execution HTTP routes and the internal dispatch contract, using generated
native TIFF inputs, real HDF5 outputs, and fake external Ray/model responses.
It covers immediate and scheduled results, replay, mask selection, empty masks,
small-slide padding, malformed model outputs, model outages, cancellation,
attempt-isolated outputs, and atomic failure cleanup. This evidence does not
establish model accuracy or real GPU compatibility.

The [review and validation record](../reviews/additional-workflows-2026-10-09.md)
records the final 235-test result, review corrections, and the matching SDK revision
required by the local environment.
