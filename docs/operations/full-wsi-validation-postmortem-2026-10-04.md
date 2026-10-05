# Full-WSI validation postmortem, 2026-10-04

This is evidence for the original deployed checkpoint. The later
[October 5 local corrections](../reviews/checkpoint-fixes-2026-10-05.md) fixed
padded-grid placement and in-memory claim ordering but were not deployed.
Preserve the measurements and hashes below as historical observations; the
corrected unaligned heatmap is not expected to match this old artifact hash.
Use [the documentation map](../README.md) for current progress.

## Summary

The deployed model-service completed both scheduled and synchronous processing of the full supplied Mirax WSI with the real prostate classifier. The scheduled execution succeeded on its first attempt in 68.03 seconds according to PostgreSQL, and the synchronous call returned in 61.61 seconds. Both produced byte-identical pyramidal TIFFs. The raw-inference SDK check returned its reference score exactly.

Each workflow covered 69,426 grid positions and selected 2,607 tiles through the tissue mask. An independent mask scan agreed with the classifier request counters. Twenty sampled heatmap values matched independently reconstructed overlap means and interpolation exactly. Eight selected classifier scores agreed with CPU ONNX execution of the cached model within an absolute tolerance of 0.002, with a maximum difference of 0.001180.

The scheduler, PostgreSQL, Ray head, autoscaler, and GPU worker remained ready with zero restarts. No OOM events or Kubernetes warning events appeared during the captured runs. The worker's recorded cgroup peak during the scheduled run was 6.51 GiB against a 24 GiB limit. No processing failure, retry, or deployment change was needed.

The orchestration and sampled numerical checks passed. Two follow-ups remain: Ratiopath slightly compresses its padded tile canvas when resizing the heatmap, and the classifier has an existing ONNX Runtime session-options keyword mismatch. This test establishes software consistency for this input; medical correctness still needs independently validated reference data.

Open the [WSI, heatmap, and overlay overview](../../downloads/validation/full-wsi-20261004/verification/overview.png), [detail overlays](../../downloads/validation/full-wsi-20261004/verification/detail-overlays.png), or [completed TIFF](../../downloads/validation/full-wsi-20261004/heatmap.tiff).

## What ran

The deployment was the existing `model-service-validation` release in `peknik-ns`, with active RayCluster `model-service-validation-jqq6z`. Both native applications were already warm and healthy. The images, dependencies, artifact selection, resource configuration, and storage layout are recorded in [the deployment document](full-wsi-deployment.md). This was a warm execution test; it did not exercise cold startup, autoscaling, or fault recovery.

The scheduled and synchronous calls used the sibling RationAI SDK against the scheduler's public service, reached through a temporary local port forward. Workflow dispatch then reached the native Ray Serve workflow, which called the classifier in the same RayService. Raw inference also used that public scheduler and classifier. No Ray Client connection or working-directory upload was used.

Input paths, checksums, and mask preparation are in [the tissue-mask record](../research/tissue-mask-for-full-slide-heatmap.md). Both executions used the full slide, with `stride_fraction=0.5`, 512-pixel model tiles, and 256-by-256 TIFF output tiles. The model target MPP of `0.468` selected WSI level 1, whose actual resolution is `0.4855947955` micrometres per pixel. No cropped or reduced inference run preceded the scheduled test.

The model file used for the independent CPU checks was the worker's already-cached ONNX artifact. Its SHA-256 was `a4db8c01735c6b4b992e1735e4bf9c1749e6e1b9dffa20b8e020cd3f0cabc3b1`. The test did not download another model from MLflow or contact an EMPAIA AppService.

## Timeline and durable state

All times in this table are UTC. Europe/Prague was UTC+2. Ray Serve's text log timestamps were seven hours behind UTC; correlation used the request ID, elapsed time, and Kubernetes log timestamps.

| Time | Event |
| --- | --- |
| 16:57:34 | Public health, readiness, and workflow-route checks passed |
| 16:57:37.994 | PostgreSQL admitted scheduled execution `2b6d9789-3bd8-45b4-a5e9-e45fb0a7cbe8`; public admission returned HTTP 202 |
| 16:57:38.068 | SDK observed `queued` |
| 16:57:38.853 | Ray workflow began processing |
| 16:57:46.158 | SDK observed `running` |
| 16:58:45.939 | Ray workflow returned HTTP 200 after 67.85 seconds |
| 16:58:46.025 | PostgreSQL recorded `succeeded` |
| 16:58:47.972 | SDK observed success, 70.31 seconds after client submission began |
| 17:09:17.697 | Zero-tile SDK check passed; synchronous full-slide call began |
| 17:10:19.305 | Synchronous SDK call returned its artifact after 61.61 seconds |

The scheduled record had one processing attempt, zero processing rejections, one standalone-provider finish attempt, no error, no lease, and no remaining work kind. Its deadline was six hours after admission. The database held 169 executions before and after the synchronous call, with zero active executions and zero unfinished provider work. The synchronous call created no durable execution record.

The scheduled request and its recovery key are preserved in [request.json](../../downloads/validation/full-wsi-20261004/request.json). Admission, status polling, and the terminal response are in the same evidence directory. Reusing this key refers to the existing execution; intentionally starting another execution requires a fresh key.

## Artifact and numerical verification

The scheduled result referenced:

```text
/mnt/validation/.workflow-attempts/208b83d82e0bc828f28f69614708ed2b39c1b8e6ae9835cd1ed12879ec54b8df/outputs/full-wsi-prostate-heatmap.tiff
```

The synchronous result referenced `/mnt/validation/outputs/full-wsi-immediate-20261004.tiff`.

Both files contained one unsigned 8-bit channel at 44,631 by 102,372 pixels, with ten pyramid levels. OpenSlide opened the completed artifact inside the worker and locally after download. The TIFF occupies 15,621,166 bytes. Both hashes were `20abd5d63f4ce0e2692a343897954803500f38f0c9a7509532c1d3419eace105`.

Verification used several independent signals:

1. A local integral-image scan of the tissue mask enumerated all 69,426 grid positions and selected 2,607 nonempty mask windows. Ray recorded 2,608 classifier calls per workflow, matching those predictions plus the model-configuration call. Two workflow calls produced 5,216 such classifier calls in total.
2. Four spatial neighborhoods across the tissue sections were evaluated with 380 separate public raw-inference requests; 104 additional candidate tiles were rejected by the ROI mask. A NumPy accumulator reconstructed overlap sums and counts without importing `SlideHeatmapProcessor` or `MaskBuilder`. The same libvips resizing operation converted those independently reconstructed neighborhoods to output coordinates. All 20 sampled TIFF values matched exactly. This checks aggregation and transport at those samples, rather than providing an independent interpolation implementation.
3. Eight selected tiles were fed to a separate ONNX Runtime CPU session using the cached model. The maximum absolute CPU-versus-deployed score difference was `0.0011797845363616943`, below the chosen `0.002` tolerance for FP32 versus the configured FP16 engine. All 380 public scores were finite and within `[0, 1]`.
4. The all-zero 512-by-512 RGB tile sent through the RationAI SDK returned `0.256591796875`, exactly matching the existing reference.
5. A streamed histogram covered every output pixel. There were 181,302,419 nonzero pixels, or 3.968% of the canvas, and values ranged from 0 to 255. Four selected background locations were zero. Histogram accumulation used strips to avoid overflowing a 32-bit histogram bin on this slide.

The output is an interpolated and quantized score map. The mask selects whole inference tiles whenever their mask window contains tissue, so the colored footprint extends beyond precise tissue boundaries. Zero also represents skipped background. The previews use blue for low scores and red for high scores; they make no diagnostic classification.

Detailed values, input-tile hashes, model hash, histogram, and coordinates are in [verification.json](../../downloads/validation/full-wsi-20261004/verification/verification.json). File equality is recorded in [artifact-comparison.json](../../downloads/validation/full-wsi-20261004/verification/artifact-comparison.json).

## Monitoring and resource observations

Monitoring began before admission. It captured scheduler, worker, head, and autoscaler container logs, plus the Serve controller, proxy, and replica log files. Public status was polled about every five seconds. Kubernetes state, cgroup CPU and memory, scratch/output file sizes, and Ray request/error metrics were sampled about every thirty seconds; namespace events were sampled about every minute. The short run consequently has only a few resource samples.

| Observation | Result |
| --- | --- |
| Scheduled worker cgroup peak | 6.51 GiB, including file cache and Ray processes, against a 24 GiB limit |
| Worker lifetime peak after additional validation and synchronous run | 6.97 GiB; cannot be attributed exclusively to the synchronous call |
| Worker CPU time between scheduled baseline and terminal sample | 297.28 CPU seconds, including background and monitoring work |
| CPU throttling | 191 additional throttled periods during the scheduled sampling interval |
| Maximum sampled scratch file bytes during scheduled run | 98,003,952 bytes, including the cached ONNX model |
| Output and scratch storage | Both retained 32 GiB PVCs stayed bound; input PVC stayed intact |
| Pods and events | All validation containers ready, zero restarts, no captured Kubernetes warning events |
| Public checks after both runs | `/healthz`, `/readyz`, and `/healthz/workflows` all HTTP 200 |

GPU process-memory queries returned `Insufficient Permissions` under the allocated MIG device. GPU utilization and peak device memory were therefore not established by this test. Namespace-scoped Kubernetes metrics were readable directly through the metrics API, while this environment's `oc top` command was unavailable. Cgroup telemetry supplies the worker resource observations above.

Ray's request counters provide coverage evidence; the model does not emit a per-tile INFO log. The logs do identify the workflow request and its HTTP 200 completion. Healthy pod status alone was not used as evidence that inference had completed.

## Problems and follow-ups

### Padded-grid registration

Ratiopath 1.4.0 constructs a 175-by-400 score grid representing a padded span of 44,800 by 102,400 model-level pixels. `resize_to_source()` resizes that entire grid to the actual 44,631-by-102,372 slide extent. This compresses the tile layout by about 0.377% horizontally and 0.027% vertically, corresponding to roughly 169 and 28 pixels toward the far edges before other interpolation effects.

The overview aligns with the tissue at thumbnail scale, and the numerical checks match this existing mapping. Pixel-exact placement needs a separate geometry regression test. A candidate correction is to resize using the tile-grid scale and crop the padded extent afterward. This behavior is in the installed Ratiopath dependency; the live validation did not alter it.

### ONNX Runtime thread settings

The [preparation record](full-wsi-deployment.md) identified `session_options=` in `models/binary_classifier.py`, while ONNX Runtime expects `sess_options=`. The configured thread settings may not apply. The runs completed successfully, but CPU throttling was observed. Correct the keyword and measure again before drawing throughput conclusions; this run does not prove that the mismatch caused the throttling.

### Download transport

The first `oc cp` of the TIFF failed with a WebSocket `unexpected EOF`, leaving 15,084,027 of 15,621,166 bytes locally. The remote workflow and artifact were already complete. Reading the same file in four bounded chunks recovered the download, and its full SHA-256 then matched both remote outputs. No workflow retry was needed.

### Reference data

There is no externally validated expected heatmap or annotated diagnostic truth for this WSI in the supplied fixtures. CPU ONNX agreement establishes consistency with the same model artifact. The slide's EMPAIA prostate label also conflicts with the scanner name recorded in the [download notes](../../downloads/wsi/962ea217-0fd7-4ddd-a1a0-165c89056cef/README.md). Tissue identity and clinical accuracy remain outside the evidence collected here.

## Retained evidence and changes

Raw evidence is under `downloads/validation/full-wsi-20261004/`, which is ignored by Git. It contains the request, SDK admission/polling/result records, logs, Kubernetes snapshots, resource samples, database checks, downloaded TIFF, numerical report, and PNG previews. Temporary local harnesses remain under `.cache/full_wsi_*.py` for inspection. They contain the exact pod and artifact identities for this run.

This session added this postmortem and a link from the deployment document. It made no application-code or Helm configuration change and published no image. Cluster changes were limited to the scheduled execution record, workflow outputs, and verification artifacts in `peknik-ns`. Temporary log readers and the local port forward were stopped; the warm deployment and artifacts remain available. Other repositories and existing worktree changes were preserved. Nothing was pushed to Git.
