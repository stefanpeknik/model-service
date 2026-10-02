# Responsibility refactor checkpoint

This implements the local architecture audit in
`responsibility-audit-2026-10-02.md`. `models/` is unchanged. No deployment,
remote execution, registry publication, or Git push was performed.

## Implemented

| Audit finding | Implementation |
| --- | --- |
| Supported inference header dropped | Public proxy preserves `x-pool-tokens`; unrelated authorization remains excluded. |
| Repository failures on polling and cancellation | Scheduler translates failures; both HTTP operations expose retryable 503 responses. Durable cancellation still precedes local cancellation. |
| Implicit input lifetime and incomplete deadline | Required `aclose`, acquisition-inclusive deadlines, bounded cancellation-resistant cleanup, and preservation of the primary failure. |
| Divergent heatmap algorithms | One slide processor with typed model metadata and an injected predictor. Ray composition is separate, and handle lookup remains lazy. |
| Catalog parsing, registration, and Helm inspection mixed | Typed declaration schemas and shared kind enums; explicit validator registration; composition and Helm validation in separate modules. |
| EMPAIA wire parsing and local storage mixed | Strict response schemas and local materialization module. HTTP authentication and mutation-outcome handling remain in the gateway. |
| Provider-owned general defaults | Prostate schema owns its defaults; EMPAIA supplies an output-destination override. Submitted values retain precedence. |
| Superseded control plane beside active code | Source, tests, container file, and Helm resources retained under `prototypes/empaia-control-plane/`; removed from active build/test/deployment paths. |
| Shared artifacts mutable across attempts | Scheduled shared-storage outputs use protected, worker-generated attempt namespaces. Synchronous destinations remain unchanged. |

Both heatmap entrypoints retain their public paths. The legacy builder is a
compatibility adapter, not a separate algorithm. Scalar classifier results,
spatial predictions, model MPP, bounded tile concurrency, and TIFF settings
remain supported. Completed task exceptions are consumed and outstanding
predictions drain before mask cleanup.

The workflow wire strings are unchanged despite internal `StrEnum` values.
Contract schemas now reject nested nonfinite JSON numbers, including malformed
HTTP requests containing NaN or infinity. Serialization remains a small
Pydantic adapter with no workflow policy.

Consumers of scheduled results must use the returned artifact path. Outputs
are no longer published to the caller's unscoped destination. No artifact
retention or garbage-collection mechanism was added. An uncooperative close
coroutine cannot be forcibly terminated; timed-out cleanup receives
cancellation and remains owned until it exits.

## Verification

- Model-service local suite: 138 tests, 123 passed and 15 skipped.
- Opt-in integration discovery: 34 skipped with all external settings disabled.
- Sibling RationAI SDK workflow suite: 13 tests, 12 passed and one skipped.
- Ruff passes for active scheduler, workflow, integration, builder, validation,
  and changed test files.
- Mypy passes for all 44 scheduler, workflow, and EMPAIA source files.
- Helm lint, offline lock consistency, EAD generated-file check, and Git
  whitespace checks pass.

Starlette's threaded `TestClient` hangs inside this execution sandbox even for
an empty FastAPI application. The same reproducer passes outside the sandbox,
where the local suite also completes. No test workaround or runtime change
was added for that environment issue.

The processor tests simulate native slide/TIFF dependencies. Real Ray, GPU,
PostgreSQL, native image output, and deployed SDK checks were not rerun. These
changes need rebuilt images and the existing opt-in deployed checks before
being treated as a deployment checkpoint. The current deployment is unchanged.

## Deferred

ONNX session extraction and the possible Virchow2 CPU bfloat16 conversion issue
would change `models/`, so both remain untouched. The bfloat16 candidate still
needs reproduction.

EMPAIA service authentication, known provider-finish failure reporting,
rectangle intersection clipping, and a storage-root/plugin authorization
policy remain follow-up work. The direct-access adapter still trusts readable
AppService-supplied paths. Its schema refactor does not provide those release
guarantees or constitute EMPAIA production sign-off.

Only `model-service` was modified in this pass. The sibling
`rationai-sdk-python` repository was inspected and tested, but not changed.
Unrelated pre-existing dependency, documentation, and demo work remains in the
working tree.
