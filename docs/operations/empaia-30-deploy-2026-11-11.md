# empaia-30 in rationai-infra-test-ns, 2026-11-11

`empaia-30` (built from `f067204`) replaces `empaia-29`: the new package layout,
raw inference removed from the API with migration `0007`, the Ray dashboard
capacity message removed, and TensorRT engines cached per checkpoint.

- `helm upgrade` to revision 12. Both API pods started on `empaia-30`; the
  database reports `0007-drop-raw-inference` applied.
- The API re-registered app `71e38adc` (standalone, preprocessing, heatmap on
  `prostate-classifier-1`), the same ID as before, so no duplicate.
- The RayService switched to the new cluster (`model-service-tp7d6`) and the old
  head was removed. `GET /healthz/deployments` answers `ready` with no missing
  routes. No GPU worker runs until a job arrives.

Not yet checked here: a job end to end on `empaia-30`, including a standalone
run now that the case browser's display fix is reported (2026-11-11). The first
job builds the prostate TensorRT engine again under its per-checkpoint path, so
expect a longer cold start. The `model-service-inference` claim is still in the
namespace; nothing mounts it any more.
