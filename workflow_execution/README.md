# Workflow execution package

This package runs a declared workflow invocation and returns a normalized
workflow result. It contains no durable execution state and no provider finish
logic.

## Start here

- `contracts.py` contains the typed invocation, input, output, result, and
  failure values, with their structural validation rules.
- `serialization.py` converts those values for HTTP transport and JSONB
  persistence using Pydantic's contract schemas.
- `submission.py` defines public standalone requests with inputs, access, optional
  destination, and an immediate-only deadline. It does not accept processing
  settings. Internal standalone invocations carry `resolved_configuration: null`;
  the workflow application uses its active catalog when each attempt starts.
  Provider-prepared and previously stored resolved configurations remain supported.
- `catalog.py` validates workflow declarations and requires configuration
  validation for every registered workflow.
- `../workflows/registry.py` composes the built-in declarations, configuration rules, and
  workflow defaults. Adding YAML without registering its configuration fails
  at startup. Each workflow ID maps to one `WorkflowConfiguration` containing
  its validator and optional defaults. Configuration-free workflows register
  `validate_empty_configuration`.
- `application.py` validates an invocation, creates input access, enforces its
  deadline, resolves deferred settings from the worker's catalog, runs the workflow,
  and validates the result. Settings are local to that attempt, not persisted back
  to the standalone queue record.
- `../integrations/shared_storage.py` implements standalone shared-storage input and output
  access and owns its strict `SharedStorageState` schema.
- `../integrations/empaia/input_access.py` materializes EMPAIA inputs without
  coupling the core workflow runtime to the provider.
  Its gateway uses the sibling EMPAIA client's transport; shared-file linking,
  rectangle-mask conversion, and consumed-response validation remain here.
- `http.py` contains the workflow HTTP error envelope shared by ingress
  adapters and the dispatch-only `WorkflowRunRequest` transport schema. Error
  codes and messages are nonempty strings, and retry decisions are strict
  booleans. The dispatch adapter validates remote failures against that same
  schema; malformed failures remain unknown dispatch outcomes.
- `dispatch.py` owns the workflow invocation interface and HTTP adapter used by
  both immediate and durable execution.
- `cleanup.py` closes resources under cancellation with a bounded wait and
  tracks cleanup tasks until they stop.

Concrete workflow code lives in [`../workflows/`](../workflows/README.md).
The heatmap workflow, processor, Ray prediction adapter, tile reader, and Serve
entry point are neighbors under `workflows/heatmap/`. The legacy
`builders/heatmap_builder.py` route still delegates to that same processor.
Deployment configuration imports `workflows.heatmap.prostate_serve:app`.

`tests/test_workflow_deployment.py` checks declarations against the enabled
Helm applications. Deployment-file inspection is test-only; the runtime uses
the separate workflow health endpoint to inspect actual Ray routes.

Workflow code receives `WorkflowContext`. It does not inspect provider
credentials, schedule records, leases, or publication state. New input systems
implement the input-access interface; new scheduling providers implement the
separate interface in `execution_scheduler/providers.py`.

`InputAccessContext` is an extensible transport envelope. Each adapter validates
its `state` once against its own strict Pydantic schema, then uses typed fields.
Shared storage accepts only a non-empty `root`. EMPAIA accepts only non-empty
`api_url` and `job_id` fields. Missing, incorrectly typed, and unexpected fields
are invalid invocations; adapter state no longer accepts credentials or
`processing_attempt_id`.

`WorkflowExecutionContext` carries the optional processing-attempt ID and
protected provider context separately. The worker supplies it only during
dispatch, without modifying the persisted invocation. Protected values remain
JSON at the encrypted-storage and generic transport seams; the EMPAIA adapter
validates them as `EmpaiaJobContext`, including a required credential and a match
against the public job location. These two context types hide credentials in
their representations, and adapter validation failures become safe error
responses. This does not add broader credential hardening to durable records.

Scheduler-to-Ray requests retain the invocation fields and add an optional
`execution_context` object. The Ray ingress passes that object separately to
`WorkflowApplication.run` and adapter `create` methods. Deploy the scheduler
and workflow images together when adopting this internal HTTP change. Direct
shared-storage requests without execution metadata remain supported and retain
their configured output paths. No SDK request or database schema change is
required. `x-processing-attempt-id` is still sent for scheduled dispatches.

Input access implements `aclose`, including a no-op for shared storage. The
invocation deadline covers input acquisition and workflow computation. Cleanup
has a separate ten-second limit and runs under cancellation. A close coroutine
that exceeds that limit is cancelled and remains tracked until it stops. If
cleanup also fails after a workflow failure, the original failure determines
the result and retry policy. Failed or cancelled acquisition releases any
resources it created before returning an access object.

Scheduled shared-storage outputs live beneath
`.workflow-attempts/<attempt-hash>/` inside the supplied storage root. The
workflow result contains this actual artifact path. Each processing attempt
therefore writes its own files, so a late attempt cannot replace an accepted
artifact. Synchronous invocations without a processing attempt identifier keep
their configured relative output paths. These files remain available for
consumers after input access closes; this module does not delete completed
artifacts.

## Native heatmap geometry regression

The slide processor reconstructs scores at the padded tile-grid scale, then
crops to the actual slide extent. It does not compress padding into the slide.
For unaligned slides this intentionally changes output pixels and hashes from
the first full-WSI validation run; inference and overlap aggregation are unchanged.

`tests/test_heatmap_geometry.py` exercises `SlideHeatmapProcessor.build` with
real Ratiopath, OpenSlide, and libvips, synthetic Aperio TIFFs, and a deterministic
prediction adapter. Enable it only in an environment with the native libraries:

```sh
RUN_NATIVE_HEATMAP_TESTS=1 UV_CACHE_DIR=.cache/uv uv run --no-sync \
  python -m unittest tests.test_heatmap_geometry -v
```

The tests check crop invariance, the original aligned-grid pixel baseline,
overlap means, dense predictions, fractional scaling, downsampling, empty masks,
and physical resolution. They require no model, GPU, or remote connection. The
local verification command using the retained image is in
[`checkpoint-fixes-2026-10-05.md`](../docs/reviews/checkpoint-fixes-2026-10-05.md).
