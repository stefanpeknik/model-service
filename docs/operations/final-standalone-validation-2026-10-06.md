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
