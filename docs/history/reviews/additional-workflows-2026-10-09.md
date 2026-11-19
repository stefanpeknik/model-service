# Additional workflow review and local validation

Status: implemented and locally validated. No images were built/published, no
Kubernetes resources changed, and no real model requests were made by this task.
Raw inference files under `models/` are unchanged. EMPAIA integration remains deferred.

Review baseline: `899f56e`. Initial implementation commit: `96ccae3`.
Two independent agents reviewed standards and the approved new workflow increment.
This report records findings and corrections, not production model acceptance.
Usage and output schemas are in [the workflow guide](../guides/additional-workflows.md).

## Standards

The review found nine documentation finding groups. New modules, classes,
helpers, fixtures, and tests lacked contracts required by `docs/guides/docstrings.md`.
Existing descriptions also omitted workflow selection, output encoding, and new
startup errors. These descriptions and new source docstrings were added.

Three judgement-call smells were reported separately:

- Duplicated complete-configuration validation. Addressed with
  `workflows/configuration.py`; schemas remain beside their concrete workflows.
  Shared TIFF geometry now lives in `heatmap/configuration.py`, so EPISeg does not
  inherit prostate-specific settings.
- Repeated small safe HTTP error envelopes in the new and existing Ray ingresses.
  Retained. Both use shared workflow error codes and the execution adapter; another
  construction abstraction would add a navigation step for two small handlers.
- Related file/stride parameters passed to embedding processing. Retained as a
  typed, single-caller function. Regrouping can follow if another caller needs it.

Mandatory documentation findings are addressed. Two nonblocking design heuristics
remain recorded above; neither changes runtime behavior.

## Spec

Four functional findings were reproduced before their fixes:

1. Embedding calls omitted the raw ingresses' trailing slash and treated HTTP 307
   as permanent failure. Requests now address `/{model_id}/`; a mounted FastAPI
   test exercises real redirect behavior rather than a permissive mock.
2. Downsampled mask windows rounded extents inward, losing selected edge pixels or
   creating zero-sized windows. Shared tile reading now rounds both edges outward.
   A real 3-by-1 mask verifies the selected right-hand tile is retained.
3. EPISeg inherited a grid that skipped slides much smaller than its 1024-pixel
   input tile. The spatial grid now includes one padded tile on short dimensions,
   matching the accumulator's existing span calculation. A 64-by-96 slide is tested
   with 1024-pixel model geometry and cropped to its original extent.
4. Plain TIFF masks depended on physical-resolution metadata. Without mask MPP
   tags, both processors use its finest level and dimension-ratio mapping. Native
   TIFF fixtures without resolution units exercise both execution paths.

All four spec findings are addressed. No model-specific scheduler logic was added.
The spec reviewer inspected the fixes and their regression tests again and found
no further regressions. That follow-up inspected code; it did not rerun the suite.
Existing prostate native geometry and cleanup checks pass. The reader corrections
preserve intended ROI selection; previously lost downsampled-mask boundary pixels
can now select tiles, so those erroneous old outputs are not golden results.

## Local validation

Final root-suite result: **235 tests passed**, in 20.077 seconds, with native image
tests enabled and disposable localhost PostgreSQL 16. Ruff passed on changed
Python files; focused strict mypy passed on 18 source files. Root discovery does
not itself run opt-in deployed integration profiles. No real Ray cluster, model
artifacts, Hugging Face model, MLflow service, or Kubernetes namespace was used.

The 15 added workflow tests use public execution HTTP routes and real internal
dispatch/application contracts, generated TIFF inputs, and real HDF5 outputs.
Only external Ray/model behavior is simulated. Coverage includes all three
scheduled workflows, replay, polling, cancellation, attempt isolation,
empty/sparse/downsampled masks, padded inputs, malformed/nonfinite features,
varying dimensions, model outages, input/output alias protection, and preservation
of an earlier completed output.

The first full run exposed the obsolete native processor fixture after the shared
helper change; its simulated native-library module now supplies that helper.
It also exposed a preexisting EMPAIA publication test injecting an HTTPX client
where the gateway requires the AppService SDK client. That fixture now bridges
the ASGI mock over loopback using the real SDK transport. No EMPAIA production
behavior was changed.

## Environment and follow-up

Sibling `../empaia` is checked out on older `master`, version 0.1.1, lacking the
shared AppService client already required by this branch. Tests used an offline
local wheel from its existing feature-branch commit
`fb18644734e0923e14b3f1b4d72ef301175f1399`, version 0.1.2, the same revision recorded
for the October 8 deployment. No sibling checkout changed. The wheel was built
from a read-only Git archive under `.cache/`; UV cache stayed in `.cache/uv`.
Running `uv sync` against the current older sibling checkout will reintroduce
that mismatch. Use the matching SDK revision for builds.

The disposable database was removed after validation. The original standalone
delivery scratchpad stays closed; this increment is recorded separately under
`~/.claude/.scratch/model-service/additional-workflows/spec.md`.

Before live testing, rebuild/retag images containing the image-local ingresses,
select matching workflow/model deployments, and provision shared storage and
model credentials. The default stock Ray head and previously published worker
images lack this new code. The full-WSI preset remains prostate-only. Validate
real EPISeg/GigaPath/Virchow2 compatibility, resource/storage sizing, and
model-owner sampling policy separately. Embeddings currently use native slide
resolution and record actual MPP, not a claimed training scale.

Summary: standards had nine mandatory finding groups and three design heuristics;
mandatory findings and one heuristic were addressed. Spec had four functional
findings, all addressed. Live model and EMPAIA acceptance are outside this change.
