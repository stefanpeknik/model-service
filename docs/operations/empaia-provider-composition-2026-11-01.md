# Composed providers beside the EMPAIA test instance, 2026-11-01

Image `empaia-24` (`94bbc94`) in [rationai-infra-test-ns](rationai-test-ns.md)
carries the provider composition, two-phase publication and worker liveness. The
Prostate cancer heatmap preprocessing app ran once from the case browser, with
the GPU worker at zero.

| Step | Time | Duration |
| --- | --- | --- |
| Admitted (execution `9cb9af86`) | 13:17:35.8 | |
| Scoring starts, after builder replica and cold GPU worker | 13:18:43 | 67 s |
| Scoring: 69,426 positions, 2,553 tissue tiles | 13:19:15.9 | 33 s |
| Grid published and job finalized | 13:19:17.8 | 1.8 s |

Total 102 s, against 101 s for the cold run on `empaia-23`. JES ended the job
`TERMINATED` 0. The GPU worker had scaled to zero when checked at 13:31.

The facade logged one `Unclosed connection` for `test-jobs-app-api`, a warning
seen on earlier runs and attributed to the EMPAIA SDK; the job was unaffected.

## `empaia-25`, 2026-11-05

`empaia-25` (`bf92155`) adds the adversarial-review fixes and drops the score
grid's region-of-interest crop, so the grid covers the whole level again. A
cold preprocessing run of the same app (execution `280c6b89`) took 106 s:
70 s until scoring started, 35 s scoring the same 2,553 tissue tiles (read
23 ms, model 69 ms per tile), and about 1.5 s to publish and finalize. JES
ended it `TERMINATED` 0.

## Registration and a standalone run, 2026-11-06

`empaia-26` (Ray) and `empaia-27` (facade) register the Prostate cancer heatmap
app themselves, with JES's service client: one app (`71e38adc`), modes
standalone and preprocessing. A facade restart re-posted it without creating a
second entry. The hand-registered `prostate_cls_preprocessing` app was deleted.

The first standalone job with a drawn region of interest (execution
`e0a3a066`) succeeded in 18 min 16 s, almost all of it a one-time TensorRT
engine build for the official batch size of 256 (about 16.5 min on the MIG
slice; the engine is now cached under `trt_cache/binary-classifier-1/b256`):

| Step | Time |
| --- | --- |
| Admitted | 01:26:50 |
| Region-of-interest mask written, model handle ready | 01:27:37 (about 40 s for the mask) |
| Model replica starts, engine build begins | 01:28:28 |
| Engine cached, scoring starts | 01:44:55 |
| Scoring: 69,426 positions, 70 tiles in the region (read 29 ms, model 62 ms) | 10 s |
| Published and finalized | 01:45:06 |
