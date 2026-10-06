# Responsibility and abstraction audit

Historical audit, not the current backlog. Read
[the resolution report](responsibility-refactor-2026-10-02.md) and
[the documentation map](../README.md) before treating a finding as still open.

Date: 2026-10-02. Source checkpoint: `68d677f`, plus the current working tree.

Three GPT-6.1-Sol agents used low reasoning effort and inherited no conversation
history. They inspected scheduler/workflow code, inference/image processing,
and integrations/prototype code separately. The coordinating agent checked
the strongest findings against their callers and ran local reproductions.
This was an exploratory design audit. Production source was not changed.

## Fixes that also clarify ownership

### 1. The inference proxy loses a supported model option

Evidence: [`inference.py:15`](../../model_service/inference.py#L15)
allows `x-output-dtype` but drops `x-pool-tokens` when the public router builds
its request. [`virchow2.py:99`](../../models/virchow2.py#L99) reads that option
to select pooled versus token embeddings. Calling the upstream model directly
and calling it through the scheduler therefore have different behavior.

A local reproduction confirmed that filtering both headers retains only
`x-output-dtype`.

Keep supported model options in one explicit inference protocol contract, and
have the proxy preserve those options. Test through the public router because
testing `HttpInferenceProxy.forward` alone bypasses the filter. Adding every
header indiscriminately would introduce a different forwarding policy.

### 2. Repository failures escape the scheduler interface on get and cancel

Evidence: [`scheduler.py:104`](../../execution_scheduler/scheduler.py#L104)
calls the repository directly for status and cancellation. Admission and
readiness translate the same repository failure into
`SchedulingUnavailableError`. The corresponding
[`HTTP handlers`](../../model_service/api.py#L307) catch only unknown
execution errors.

A local HTTP reproduction with a repository raising
`ExecutionRepositoryUnavailableError` returned unstructured 500 responses for
both polling and cancellation.

Make the scheduler's error contract consistent across its public operations,
then translate it to the existing retryable HTTP envelope. Keep local task
cancellation after the durable cancellation succeeds. Test outages through
both public endpoints.

### 3. Input-access lifetime is implicit, and the deadline covers only computation

Evidence: [`WorkflowApplication.run`](../../workflow_execution/application.py#L79)
acquires input access before opening its timeout scope. Cleanup is discovered
with `getattr(access, "aclose", None)` rather than declared by the input-access
interface. The EMPAIA adapter performs asynchronous materialization during
[`create`](../../integrations/empaia/input_access.py#L71).

Local reproductions confirmed two defects:

- Input acquisition continued beyond a 5 ms invocation deadline and required
  an external 50 ms timeout to stop it.
- A permanent workflow failure followed by a cleanup `OSError` became a
  retryable `workflow_internal_error`. Cleanup replaced the primary exception.

There is also a cancellation ownership gap in the EMPAIA factory. Its
[`except Exception`](../../integrations/empaia/input_access.py#L100) cleanup
does not handle task cancellation while acquiring access.

Give input access an explicit managed lifetime. An async context-manager
interface would cover acquisition and release with one ownership convention;
a required async close operation with a shared-storage no-op is a smaller
alternative. Apply the absolute deadline to acquisition and computation,
preserve the original failure if cleanup also fails, and give cleanup a bounded
lifetime. Retain the existing distinction between an actual expired deadline
and an adapter raising `TimeoutError` for another reason.

Tests should cover slow acquisition, cancellation during acquisition, and
combined processing/cleanup failures. Existing cleanup tests cover failure
after successful processing, which misses exception precedence.

### 4. Two supported heatmap paths implement different algorithms

Evidence: [`HeatmapBuilder`](../../builders/heatmap_builder.py#L54) and
[`RayProstateHeatmapProcessor`](../../workflows/heatmap/prostate.py#L141)
repeat slide selection, tissue scaling, tile scheduling, prediction
reconstruction, mask aggregation, and TIFF writing. Both applications appear
in the default Helm application list; this audit did not inspect a live
deployment.

The implementations have already diverged:

- The legacy builder accepts only 2D/3D predictions. The
  [`binary classifier`](../../models/binary_classifier.py#L135) returns a
  scalar for an individual batched request. That becomes a 0D NumPy array and
  fails the builder's shape check. The newer processor reshapes using model
  metadata and handles the scalar case.
- The newer processor clamps stride to at least one and cancels/drains pending
  tile tasks before cleanup. The legacy builder does neither. Pending tasks
  can continue using mask resources after another task fails and cleanup runs.

Extract one slide-processing implementation, retaining both public entrypoints
as adapters. Supply prediction and validated model metadata to the core rather
than resolving a Ray application inside the algorithm. Construct the Ray
adapter in the deployment composition. This also removes the need for the
processor test to substitute several image/Ray modules in `sys.modules`.

Preserve scalar and spatial prediction handling, model MPP, TIFF layout,
bounded task concurrency, and cancellation. Keep temporary-file publication in
`ProstateHeatmapWorkflow`. Source tracing establishes the scalar mismatch;
the production model and image libraries were not loaded for this audit.

## Further simplification targets

### 5. The workflow catalog combines declarations, implementation registration, and Helm inspection

[`WorkflowCatalog.default`](../../workflow_execution/catalog.py#L73) imports
the concrete prostate implementation to register its configuration validator.
[`validate_configuration`](../../workflow_execution/catalog.py#L114) silently
accepts a workflow without a registered validator. Adding a YAML declaration
therefore does not necessarily add admission validation.

The same module manually parses declaration fields and knows how to inspect
[`Helm application files`](../../workflow_execution/catalog.py#L128), although
that inspection is used by a consistency test.

Use typed declaration schemas for structural parsing. Register executable
workflows and their configuration schemas together during composition, with
an explicit configuration-free schema where appropriate. Move deployment-file
inspection to deployment validation. Keep the catalog's caller interface small.
Share closed input/output kind definitions with runtime contracts instead of
repeating literal unions and string-to-kind functions.
Missing registration is an extension risk; the current default prostate
registration exists.

### 6. EMPAIA wire parsing and materialization need clearer internal ownership

[`HttpEmpaiaGateway`](../../integrations/empaia/http_gateway.py#L102) handles
HTTP, handwritten response validation, configuration merging, direct-access
symlinks, rectangle conversion, and output document construction. Its small
public interface is useful and should remain.

Start with typed AppService response schemas. This replaces repeated mapping,
string, and numeric checks and gives response validation one owner. Keep the
EMPAIA-specific semantic mapping behind the gateway interface. Isolate local
materialization where storage policy needs to vary; splitting every gateway
method into another module would add navigation costs without useful depth.

A concrete parsing defect illustrates the current drift:
[`rectangle coordinates`](../../integrations/empaia/http_gateway.py#L364)
are not checked for finiteness, while widths/heights are. Local calls with NaN
and infinity produced raw `ValueError` and `OverflowError` rather than
`EmpaiaGatewayError`. A finite-coordinate schema would centralize the missing
constraint. Test malformed responses through input materialization and retain
the existing safe error contract.

Preserve mutation guards and the distinction between safely retryable failures
and unknown publication outcomes throughout any extraction.

### 7. Workflow defaults are owned implicitly by one provider

[`EmpaiaConfigurationRegistry.default`](../../integrations/empaia/configuration.py#L21)
imports the concrete prostate configuration and chooses its stride, tile
dimensions, and output path. The provider applies those defaults before
catalog validation. Standalone callers provide resolved configuration directly.

Put general workflow defaults alongside their configuration schema. Represent
provider-specific values, such as an EMPAIA output destination, as explicit
overrides supplied during composition. Preserve current override precedence
and path behavior; making both providers identical without considering storage
semantics would be a behavior change.

### 8. ONNX setup is duplicated inside inference deployments

[`BinaryClassifier`](../../models/binary_classifier.py#L44) and
[`SemanticSegmentation`](../../models/semantic_segmentation.py#L42) duplicate
artifact loading, cache construction, TensorRT profiles, execution-provider
ordering, session options, and metadata discovery.

A shared ONNX session factory can own runtime configuration and return typed
model metadata. Keep Ray batching and model-specific output handling in their
deployments. Avoid an inheritance hierarchy. Validate the assumed input name
and channel layout instead of configuring them independently of discovered
metadata. Preserve provider ordering and profile/cache settings, then check
numerical behavior with the existing deployed classifier golden test.

### 9. The superseded control plane remains alongside the active implementation

[`The prototype document`](../architecture/empaia-control-plane.md#L3)
explicitly marks `control_plane/` as superseded, its flow ends at `READY`, and
its Helm option is disabled by default. Its SQLite transitions, serialization,
configuration validation, and lifecycle terminology duplicate active concerns.

Archive it with its tests and deployment references, or make its historical
status visible at the source entrypoint. Schema migration and transition
extraction there have low value unless the prototype is revived. Do not
introduce it as a second active scheduler or copy its full implementation into
the current adapters.

## Policy choices to resolve before changing behavior

- **Artifact identity.** Shared-storage results reference a caller-selected
  path, and [`publication replaces that path`](../../workflows/heatmap/prostate.py#L63).
  Different executions or overlapping remote attempts can overwrite the same
  file. Database fencing protects settlement, not filesystem writes. Decide
  whether scheduled results should reference immutable attempt artifacts or
  intentionally mutable caller destinations. An immutable internal artifact
  plus an explicitly managed caller destination is one option. This changes
  path expectations and needs concurrency tests.
- **AppService path trust.** The active gateway symlinks an AppService-supplied
  readable path without an explicit allowed-root/plugin policy. The prototype
  had such a policy. If AppService paths are deliberately trusted, document
  that assumption; otherwise inject a storage resolver into materialization.
  A stricter resolver needs a deployment migration and path/plugin tests.

An additional model-runtime candidate needs reproduction: Virchow2 can use CPU
bfloat16 autocast, then converts its result to NumPy before casting to the
requested output dtype at [`virchow2.py:112`](../../models/virchow2.py#L112).
This conversion order should be checked with an actual bfloat16 result before
treating it as a confirmed production failure.

## Abstractions that earn their place

- `ExecutionTransitions` keeps lifecycle decisions consistent between the
  PostgreSQL and in-memory repositories.
- Transactions, fencing, lease recovery, database-clock deadlines, and provider
  mutation tracking implement actual durability guarantees.
- Workflow dispatch, provider adapters, and workflow input access represent
  genuine implementation differences.
- `WorkflowApplication` usefully owns execution policy; its resource lifetime
  needs tightening.
- Artifact publishers and rectangle writers isolate image-library dependencies.
- Temporary output files and atomic publication prevent partial completed
  artifacts, even though overwrite ownership needs an explicit policy.

Suggested sequence: correct the public inference/error contracts and input
lifecycle first; consolidate heatmap processing next; then improve declaration
and EMPAIA schemas/composition. Treat prototype archival and artifact/storage
policy as separate changes with their own compatibility decisions.
