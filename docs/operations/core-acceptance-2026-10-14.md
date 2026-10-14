# Core acceptance in peknik-ns, 2026-10-14

Live checks of the builder + model catalog core against the acceptance list in
[the specification](../spec.md). Local test results are separate: 250 unit and
PostgreSQL-backed tests passed at the same revisions.

## What ran

| Item | Value |
| --- | --- |
| model-service | branch `claude/model-service-core`, commits up to `9a24557` |
| Ray image (head, CPU and GPU workers) | `cerit.io/stefanpeknik/model-service-workflow-gpu:core-b9f7997`: a code layer (plus `h5py`, `timm`) over the October 8 GPU image |
| Facade image | `cerit.io/stefanpeknik/model-service-scheduler:core-eb93a32` |
| Helm | release `model-service-validation`, revisions 10–14, `values.peknik.yaml` |
| Workers | one CPU worker (4 CPU, 8 GiB) for builders and the constant model; one MIG `2g.20gb` GPU worker, minimum 0 |
| Slide | Mirax prostate WSI `67df0411-…mrxs` (89262 × 204744 at 0.243 µm/px) with its tissue mask, SHA-256 `64cb47ee…` |
| SDK | `rationai-sdk-python` branch `claude/builders` |

All requests went through the public facade (`kubectl port-forward` to
`model-service-validation-execution-scheduler:8080`).

## Results

| Check | Result |
| --- | --- |
| Migration `0002` on the existing database | 186 terminal `prostate-heatmap` rows became `heatmap` / `prostate-classifier-1` |
| `GET /v1/models`, `/v1/builders`, `/healthz/deployments` | Catalog matched the Helm model files; all builder and model routes present |
| `heatmap` + `constant-classifier` (CPU only), synchronous | Succeeded; 9 min 15 s (128-px tiles, two CPU threads) |
| `heatmap` + `prostate-classifier-1`, synchronous, cold GPU | Succeeded in 3 min 1 s including GPU worker start and model load; TensorRT cache reused |
| Same, compared with the previous implementation | Codex's processor at `232412f`, run in the same cluster against the same model and inputs, produced a pixel-identical TIFF (maximum difference 0) |
| Scheduled run, facade pod deleted while running | Succeeded after a second processing attempt; output pixel-identical to the synchronous run |
| Cancellation during a scheduled run | `cancelled`; no artifact left; the builder replica was free immediately afterwards |
| YAML-only onboarding | Virchow2 added as `models/peknik/virchow2.yaml` plus one list entry; no Python change, same image tag; listed in `/v1/models` after `helm upgrade` |
| `tile-embeddings` + `virchow2` | Succeeded in 3 min 41 s cold; slide level 1 (0.486 µm/px) chosen for the declared 0.5; 3,078 tiles × 2 summaries × 1,280 floats, all finite |
| Two models at once | A heatmap and an embedding run executed concurrently; a second heatmap waited for the first (per-model limit 1) |
| RationAI SDK | `models.list`, `builders.list`, raw `classify_image`, `schedule` with replay, `executions.get` and `cancel`, typed `IncompatibleModelError` |
| GPU release | GPU worker removed 6 min 24 s after the last request |

## Findings fixed during the run

- A non-slide `slide` input produced a retryable internal error. Unreadable
  slides and masks are now invalid input (`b0f9825`).
- Builders can be placed on any worker with free CPU, so the GPU worker group
  now mounts the shared and scratch volumes too.
- The October image lacked `h5py` and `timm`; the refresh image installs them.
- A stale `INFERENCE_SERVE_ORIGIN` left by an earlier `kubectl set` pointed raw
  inference at the old release. Recreating the Deployment removed it.

## The October 8 reference file

`native-full-wsi-0eca8b5d…-immediate.tiff` differs from today's output in
57.8 M pixels (maximum 122/255). The previous implementation, run today on the
same inputs, reproduces today's output exactly, so that file was not produced
from these inputs and settings and is not a valid golden reference. Its origin
is not recorded.

## Not established

- EMPAIA: no AppService, JES or EMPAIA job was involved.
- EPISeg: its MLflow artifact is not reachable from this namespace.
- Clinical correctness, multi-GPU scaling, or load beyond one GPU.
