# Dead-code audit

Historical audit with its resolution recorded below. Use
[the documentation map](../README.md) for current module ownership and progress,
not this candidate list as a new removal task.

Source: `009426d` and the current working tree, 2026-10-02.
Two fresh GPT-6-Luna agents used high reasoning effort and inherited no
conversation history. One inspected the scheduler and deployment/support
code; the other inspected workflows, EMPAIA, and the sibling RationAI SDK.
The coordinating agent cross-checked candidates against callers and
composition. The audit made no source changes and ran no tests or deployments.

## Confirmed removal candidates

### Catalog deployment validation left behind

`workflow_execution/catalog.py:72-78,164-167` defines
`MissingWorkflowApplicationsError` and `WorkflowCatalog.validate_deployed`.
Their only caller is `tests/test_workflow_catalog.py:110-116`, which tests the
method against an empty set. No runtime composition invokes it.

`tests/test_workflow_deployment.py:9-23` now reads Helm configuration directly
and checks that declared workflow IDs are enabled and have matching Ray
application names. The old method and exception were left behind when their
production-module caller was removed. Remove both and their self-test; retain
the actual repository consistency check. Confidence: high.

### Unread claim-copy fields

`execution_scheduler/domain.py:129,132` declares `WorkClaim.deadline_at` and
`WorkClaim.lease_expires_at`. `execution_scheduler/transitions.py:169,172`
populates them, but no consumer reads either field.

The worker sends `claim.invocation`, which already contains the deadline.
Repository state, database clocks, and the configured lease duration own
renewal and fencing. Remove the two claim fields and constructor assignments.
Do not remove the corresponding `ExecutionRecord` fields or persistence
columns, which are live. Confidence: high.

### Unused direct-access identity copy

`integrations/empaia/http_gateway.py:30,207` declares and populates
`DirectAccess.slide_id`. Local materialization consumes only `storage_path`
and `supported`; no code reads this internal identity field.

Remove the internal field and assignment. Keep the separate wire-schema
requirement in `DirectAccessResponse` pending an AppService contract decision.
The local validation AppService emits that field, but local evidence does not
establish whether accepting responses without it is allowed. Confidence:
high for removal of the internal copy.

### Redundant default EMPAIA override

`integrations/empaia/configuration.py:20` supplies
`output_path="heatmap.tiff"`, identical to the workflow default at
`workflow_execution/prostate_heatmap.py:69`. Provider preparation applies this
through the catalog's existing defaults/overrides/submitted-values resolution.
The default entry therefore changes no result.

Remove the duplicate default entry. The registry itself has no custom
instantiations in the inspected repositories, so its continued need should be
evaluated when implementing this cleanup. Its methods are called at runtime;
do not describe the whole module as unreachable. Confidence: high for the
redundant entry, not for blanket deletion of the override mechanism.

## Stale value with a public-schema caveat

`execution_scheduler/api.py:54` declares
`ApiErrorCode.WORKFLOW_APPLICATION_UNAVAILABLE`. No scheduler route emits the
member or its string value. The actual workflow-unavailable code is
`"workflow_unavailable"`; synchronous dispatch forwards workflow failure
codes as strings, and its error schema accepts a string.

The unused member is nevertheless included in OpenAPI through
`SchedulerErrorDetail.code`. Removing it changes the published enum. It is a
stale value, but deletion should be an intentional contract cleanup rather
than an assertion that it has no external visibility. Confidence: high on
absence of emission; external client dependency is unknown.

## Test-only code that is not dead

- `InMemoryExecutionRepository` and `InMemoryWorkflowDispatch` are used broadly
  by local and integration test harnesses. Production uses PostgreSQL and HTTP
  dispatch. Moving the fakes is optional organization work, not a reason to
  delete their behavior.
- `validate_empty_configuration` is currently used only by test workflows.
  The package README also documents it as an extension utility. Either retain
  it explicitly as supported utility code or move the helper with the test
  fixtures; absence of a shipped configuration-free workflow is not proof of
  unreachable code.

## Verified entrypoints and exclusions

The agents traced Docker ASGI commands, Helm `import_path` declarations,
FastAPI decorators, Ray `.bind()` entrypoints, dynamic gateway factories,
model-provider `_target_` declarations, CLI scripts, demo instructions, and
SDK client/type exports. These were not treated as dead because they lack
ordinary Python import callers. Validation profiles and recovery-control
routes have documented deployment/test uses.

No safely dead SDK API was identified. The legacy heatmap route remains used
by the SDK's public slide resource. `models/` was not modified, and the
explicit historical archive under `prototypes/` was excluded from active
dead-code findings. No external reads, writes, pushes, or deployed tests were
performed.

This is a static caller and composition audit. It identifies concrete cleanup
candidates, not proof that arbitrary external clients never use public
Python symbols or OpenAPI values.

## Resolution

The coordinating agent verified the findings and implemented the cleanup in
`model-service` only. The source locations above refer to the audited revision.

- Removed the unused catalog deployment validator, its exception, and its
  self-test. Kept the Helm consistency test in `test_workflow_deployment.py`.
- Removed the unread deadline and lease copies from `WorkClaim`. Durable
  execution state, database columns, renewal, and invocation deadlines are
  unchanged.
- Removed the internal `DirectAccess.slide_id` copy. The AppService response
  schema still requires and validates `slide_id`.
- Removed `integrations/empaia/configuration.py` and its provider constructor
  hook. No inspected caller supplied a custom registry. Its sole default
  duplicated the workflow default, so EMPAIA now resolves configuration through
  the catalog directly. Removed the otherwise unused catalog `overrides`
  parameter. Submitted values still override workflow defaults and undergo the
  same validation. The deleted module remains recoverable in Git.
- Removed the never-emitted `workflow_application_unavailable` API enum value.
  This intentionally narrows the published OpenAPI enum. A before/after schema
  comparison confirmed that this is the only OpenAPI change; endpoint paths,
  other schemas, and emitted error codes are unchanged.

Retained the in-memory test implementations and the documented
`validate_empty_configuration` utility. No SDK changes were needed. `models/`
and unrelated working-tree changes were left untouched.

### Validation

- Focused workflow, scheduler, HTTP, EMPAIA, resource, and retry suites passed
  all 102 tests.
- Full local discovery ran 141 tests: 126 passed and 15 skipped. External-test
  opt-ins were disabled. FastAPI TestClient required running outside the
  sandbox because of its thread/event-loop restrictions.
- Integration-test discovery collected 34 tests, all skipped with external
  opt-ins disabled. No deployed behavior was retested in this cleanup.
- Ruff lint passed for active scheduler, workflow, integration, builder, and
  validation packages and the changed test. Formatting passed for changed
  Python files. Strict mypy passed for 42 scheduler, workflow, and EMPAIA files.
- Helm lint, offline lockfile validation, and `git diff --check` passed.

No deployment, remote write, or push was performed.
