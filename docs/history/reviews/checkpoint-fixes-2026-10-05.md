# Checkpoint regression fixes, 2026-10-05

These are local corrections after the successful full-WSI checkpoint. No image
was published and no deployment was changed. The warm deployment and the
2026-10-04 artifact remain the historical baseline, not evidence for this code.

## Persistence claim policy

The in-memory adapter previously selected eligible records only by creation
time. PostgreSQL prioritizes provider finish and terminal reports, then processing;
processing workflows are considered by oldest eligible creation time, and records
within a workflow by due time followed by creation time.

The correction changes only in-memory selection. PostgreSQL queries, locking,
transactions, state transitions, retry limits, and fencing are unchanged.
Eight shared contract scenarios now run against both adapters. They cover finish
and report priority, due-time order, workflow ordering and capacity, future retry
eligibility, and cancellation fencing. Existing PostgreSQL-specific concurrency,
deadline, settlement, and lease-recovery tests remain.

The initial provider-finish priority scenario failed on the in-memory adapter
before the correction. Both adapters now pass it. PostgreSQL verification used a
disposable local container and a loopback-only connection, not the deployed database.

## Heatmap placement

Ratiopath's `resize_to_source()` scaled its padded accumulator to the unpadded
slide dimensions. The processor now scales to the actual span of the tiled canvas,
then crops the right and bottom padding. Aggregation, interpolation kernel,
quantization, TIFF settings, and cleanup remain unchanged.

A synthetic 107-by-75 slide with overlapping 32-pixel tiles reproduced the
error. Its output differed at 4,200 of 8,025 pixels from the corresponding crop
of the 112-by-80 aligned slide. The same comparison now matches exactly.

Native tests cover multiple strides, scalar and dense predictions, fractional
scaling and downsampling, constant overlap means, empty tissue masks, dimensions,
and physical resolution. The aligned-grid pixel SHA-256 was captured from the
unchanged retained image before the fix and remains exactly
`01a90dd369c6c5b5122335061de7cc609156cb094798a5d3851cc98a2333b58f`.

Unaligned output changes intentionally. The old full-WSI TIFF hash is not an
expected hash after deploying this correction. A future deployed run should
repeat artifact inspection and geometry checks; no full-WSI model execution was
performed during this correction.

## Verification

- Local suite with isolated PostgreSQL: 180 tests run, 175 passed, 5 skipped.
  Four skips are the native heatmap profile; one requires libvips. No
  deployed-test opt-in was enabled.
- Offline native image: 8 tests passed, including the four geometry tests,
  three existing processor tests, and TIFF publication to an in-process mock
  AppService. All five tests skipped locally passed in this native run.
- Ruff checks and formatting passed for all seven changed Python files.
- Mypy passed for `execution_scheduler/repository.py` and
  `workflow_execution/heatmap.py` with unrelated imports followed silently.

Reproduce the native checks from the repository root using the retained image:

```sh
podman run --rm --pull=never --network=none --read-only \
  --cap-drop=ALL --security-opt=no-new-privileges --security-opt label=disable \
  --tmpfs /tmp:rw,size=512m -e RUN_NATIVE_HEATMAP_TESTS=1 \
  -v "$PWD/tests:/app/tests:ro" \
  -v "$PWD/workflow_execution:/app/workflow_execution:ro" \
  -v "$PWD/validation:/app/validation:ro" \
  cerit.io/stefanpeknik/model-service-workflow-gpu:full-wsi-20261002-1 \
  uv --no-cache run --no-project python -m unittest \
  tests.test_heatmap_geometry tests.test_prostate_heatmap_processor \
  tests.test_validation_empaia_publication -v
```

The ONNX Runtime keyword mismatch remains pending permission to modify
`models/`. Broader validation consolidation and scheduler fixture cleanup were
not included in these targeted fixes.
