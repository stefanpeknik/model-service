# Workflow execution proposal

Status: work in progress, last reviewed 2026-09-16. This records the current
design discussion, not an approved implementation specification. Update this
page when a decision changes; do not treat the exploratory
archived `prototypes/empaia-control-plane/` code or notes 16 and 17 as the source of truth.

The small module interfaces and durable work model for the first generic slice
are settled in [`execution-interfaces.md`](execution-interfaces.md).

## Goal

Keep model-service useful on its own through the RationAI SDK while allowing
EMPAIA to submit whole-slide work without a separate per-EMPAIA-job adapter pod. A
single workflow implementation should process a WSI or ROI whether the caller
waits for it or schedules it. EMPAIA integration supplies EMPAIA-job-specific input,
output, progress, and authorization handling; the execution scheduler does not
interpret EMPAIA jobs.

This page distinguishes three things that earlier plans mixed together:

| Concern | Choices in scope | What it controls |
| --- | --- | --- |
| Work | Model inference or workflow | Tile prediction versus whole-slide processing |
| Invocation | Synchronous or scheduled | Open request versus durable execution record |
| Data integration | Standalone filesystem or EMPAIA | How inputs and outputs are accessed |

The chosen invocation mode does not dictate the data integration. The immediate
EMPAIA use case is scheduled only, but the RationAI SDK should expose both modes
for workflows.

## Agreed decisions

### Work and invocation

- `models/` retains synchronous, model-ready inferences. Its existing HTTP
  routes accept tiles, not WSI paths.
- A workflow owns tiling, calls model inferences, and produces a useful result
  for a whole WSI or ROI. One implementation serves synchronous and scheduled
  invocations. A scheduled record represents the entire workflow, not each
  tile or model inference.
- A workflow result is a named output map matching the workflow's declared
  output contract. Most workflows may have one entry, while a workflow may
  return several heterogeneous outputs. Each entry is either an inline
  JSON-compatible value or an explicit shared-filesystem artifact reference;
  an ordinary string is never guessed to be a path. The execution scheduler
  must persist the normalized map before retryable provider finish.
- A scheduled standalone caller polls execution status and can retrieve the
  workflow result after success. The durable execution stores the normalized
  output map rather than relying on a live Ray actor. The first slice does not
  automatically delete execution records or generated artifacts.
  Retention periods, ownership-aware cleanup, and storage limits are deferred.
- Synchronous means the caller waits for a response; it is not a real-time
  latency promise. Scheduled means model-service commits a durable execution
  record and returns an execution ID. A `202 Accepted` response follows the
  commit, not the start of inference.
- Synchronous and scheduled workflow invocation are separate public HTTP
  operations, not a `mode` field on one operation. They still call the same
  programmatic workflow implementation. Their operation names are `run` and
  `schedule`; exact URI paths are deferred.
- Model-service presents one logical public origin to the RationAI SDK and
  JES. Deployment routing sends synchronous model and workflow operations to
  Ray Serve, while scheduling, status, cancellation, and
  provider-specific operations go to the execution scheduler. A synchronous
  invocation does not pass through scheduled orchestration even though both
  are reached through the same model-service origin.
- A schedule request is idempotent. The RationAI SDK generates an idempotency
  key and reuses it for transport retries; callers may supply one when they
  need recovery across client restarts. Reusing a key returns the existing
  execution instead of creating duplicate work. An EMPAIA job ID fills the
  same role for EMPAIA schedule requests.
- The first public status model stays coarse: `queued`, `running`, `succeeded`,
  `failed`, and `cancelled`. Retry waits and provider finish do not become
  separate public phases yet.
- A failed execution exposes a stable error code and safe message. Raw
  exceptions, credentials, and provider response bodies stay in protected
  logs rather than the public execution record.
- A synchronous workflow stops when its HTTP caller disconnects. It does not
  become a scheduled execution implicitly.
- Ray Serve's request queue is not durable execution storage. Accepted work
  must survive ordinary Ray, execution-scheduler, and model-service restarts.
- After a scheduled worker failure, model-service may rerun the whole workflow.
  The first version does not need mid-workflow recovery. Provider finish
  must tolerate a repeated attempt.
- Processing retries should use a simple rule: do not retry failures known to
  be permanent, such as invalid input. Errors whose outcome is uncertain may
  retry with capped exponential backoff up to a configurable attempt limit.
  The default values and exact permanent-error cases still need definition.
- Standalone scheduled executions have a configurable default deadline rather
  than waiting indefinitely. A caller may request a timeout within the
  deployment's limit. The timeout covers queue wait, processing, and provider
  finish.
  EMPAIA's caller may choose its own requested timeout, but model-service
  caps the effective deadline at the original token's actual expiry, not an
  assumed fixed 24-hour interval.
- Scheduled work should leave best-effort capacity for synchronous calls. No
  latency guarantee is made. The execution scheduler applies a configurable
  per-workflow limit to outstanding scheduled requests. The first version
  does not derive that limit from Ray autoscaling settings; a synchronous
  request may still wait behind work already running.
- The first scheduled workflow-dispatch adapter uses HTTP rather than a Ray
  deployment handle. A local Ray 2.55.1 probe found that client-process loss
  cancelled the HTTP request while handle work continued. HTTP cancellation
  is still best effort, so database leases and attempt fencing remain
  required. Start with one outstanding scheduled request per workflow and
  enforce that limit in the durable claim transaction.
- A valid scheduled invocation is accepted even when its registered workflow
  has zero active replicas, no current GPU capacity, or temporarily unavailable
  Ray capacity. It remains in the durable queue until the execution scheduler can
  dispatch it within the per-workflow limit. An unknown workflow is rejected;
  a declared contract with no corresponding Ray application is a deployment
  health error. Synchronous invocations continue to depend on the caller's
  open request and may time out while capacity starts or becomes available.
- The RationAI SDK keeps its model-inference methods and gains `workflows.run`,
  `workflows.schedule`, `executions.get`, and `executions.cancel`. Exact HTTP
  paths and payloads are open.
- The first standalone workflow and execution endpoints add no authentication
  or per-user execution ownership. They must remain inside a trusted
  deployment. EMPAIA scheduling keeps its separate service authentication.
- The execution scheduler owns scheduled orchestration: claim an execution,
  invoke its Ray workflow, persist the workflow result, run provider finish,
  and update terminal status. This describes responsibility, not one large
  class. Repository access, dispatch, provider handling, and state transitions
  may remain separate modules behind that orchestration.
- Ray workflow and model actors do not access the execution database. The
  execution scheduler passes resolved configuration and receives small results or
  shared-storage paths; decoded WSI tiles do not pass through it.
- Execution-scheduler instances use ordinary database-backed leases, heartbeats, and
  attempt fencing so another instance can recover abandoned work. The first
  deployment may use one execution-scheduler replica, but the design does not assume
  a permanent singleton.
- The execution scheduler is a long-lived deployment separate from the Ray
  Serve
  workflow and model applications. Submitting an execution does not create a
  per-execution scheduler or adapter pod.

### Data integration

- Standalone whole-slide workflows require shared filesystem access to their
  WSI inputs. File-producing workflows initially use caller-supplied output
  paths on shared storage, with no HTTP artifact download. A scalar-producing
  workflow need not write an output file; it returns the value in the same
  normalized named-output map.
- An EMPAIA schedule request carries EMPAIA job context, not a translated WSI
  filesystem path. No WSI or decoded tile data passes through JES, the
  execution scheduler, or the
  schedule request. The Ray-side workflow process constructs the
  EMPAIA client from the EMPAIA job context and credentials and uses it to read its
  inputs. The client may later redirect HTTP tile access to a filesystem
  internally; model-service must not implement or depend on that transport.
- A standalone RationAI SDK workflow call supplies paths on shared storage, and
  the Ray-side workflow opens those paths. Shared workflow utilities may hide
  differences between filesystem and EMPAIA-backed inputs, but they do not
  introduce another WSI transfer protocol. Serialization of tiles between a
  Ray workflow actor and model actors is a separate internal optimization.
- Provider-specific parsing and configuration resolution happen before the
  shared workflow. The workflow uses supplied input-reading, progress, and
  cancellation capabilities to produce a result. The execution scheduler then
  calls the selected provider adapter to finish the outcome. The workflow does not
  parse an EAD, fetch EMPAIA job configuration, or call EMPAIA publication.
- The EMPAIA-facing scheduling route and the RationAI SDK scheduling route
  should enter the same execution-scheduler interface after their request-specific
  validation. The execution scheduler does not interpret EMPAIA job context.

### EMPAIA lifecycle

- One EMPAIA job triggers one scheduled execution. Model-service durably
  records their association and uses EMPAIA job identity for idempotent
  admission. A repeated schedule request returns the existing execution without
  overwriting its configuration or state. JES remains the EMPAIA execution
  registry; model-service owns its accepted execution.
- For a new EMPAIA schedule request, the provider adapter fetches and validates
  its EMPAIA job configuration before acceptance. Model-service stores the resolved
  values with the durable record, then returns `202`. A fetch or validation
  failure returns an error without creating an execution. Exact HTTP error
  codes and JES retry behavior are deferred.
- The future JES `ModelServiceClient` submits the EMPAIA job, polls model-service
  through the existing `await_job` contract, and forwards cancellation. No
  callback from model-service to JES is planned.
- Model-service uses only the original EMPAIA job token. It does not mint or renew a
  token. Token expiry is a hard execution deadline with no separate margin.
  The token must be stored securely because accepted work survives restarts.
- For EMPAIA success, model-service awaits all required upload and finalize
  library calls. Successful returns are enough; it does not make a separate
  confirmation poll. Transient publication errors get a small, bounded number
  of retries with backoff, approximately three. If publication still cannot
  be confirmed, model-service fails the execution. A lost successful response
  may leave the external outcome ambiguous; duplicate-call behavior must be
  checked before implementing these retries.
- EMPAIA publication is a separate step of the same scheduled execution after
  workflow processing. It retries from a retained result without repeating
  model inference. That result must survive worker restarts while publication
  remains pending, whether it is a small value or a large artifact; the
  publication step does not require another pod.
- After publication retries are exhausted, model-service attempts to report
  failure to EMPAIA while the original token remains valid. The local
  execution becomes failed even if that failure-reporting call also fails.
- The same best-effort EMPAIA failure report applies to any terminal failure
  after admission, including a processing failure. Cancellation stops active
  processing or publication as soon as practical. A completed EMPAIA
  finalization cannot be undone; exact cancellation races are deferred.
- JES backend selection may later become a small capability-based JES change.
  That work is deferred. An HTTP-looking image value is not the agreed routing
  contract.

### Configuration

- Each model deployment's Ray application YAML owns its model artifacts, tile
  size, batching, resources, and target MPP where relevant. A workflow is not
  constrained to one such deployment: its programmatic implementation may
  invoke zero, one, or several model inferences and may reuse ordinary Python
  workflow components.
- A workflow definition owns the generic workflow identity and input/output
  contract. Its optional EMPAIA section generates an EAD rather than
  duplicating that contract in a second handwritten file. It does not select a
  single model or a reusable Python helper through top-level YAML
  fields.
- The stable workflow identifier is also the Ray Serve application name. The
  execution scheduler dispatches by that name, while the application's Python
  code defines its internal sequence of model inferences. A build or startup
  check should detect a declared workflow contract without a corresponding
  deployed application. This convention does not require the public HTTP route
  to mirror Ray's internal route.
- Internal routing such as the model-service origin, concrete model
  deployments, workflow selection, and resource profile must not be
  EMPAIA-configurable EAD values.
- Accepted executions use a validated, immutable snapshot of their concrete workflow
  values. Configuration-specific database columns are not required; JSONB is
  acceptable. The registered app identity ties EMPAIA values to its EAD
  contract.
- An execution names a workflow but does not pin its implementation or model
  revision. It runs the currently deployed implementation for that stable
  workflow contract. Backward-compatible changes are a deployment
  responsibility in the first version.
- PostgreSQL is the production durable execution store. A narrow repository
  interface keeps scheduling logic independent of PostgreSQL and supports
  focused test adapters; it is not a general pluggable-database framework.
- EAD registration is deferred. A future reconciler near the deployed service
  is favored over CI/CD because the service has the relevant identity and
  network access.

The draft prostate example lives in
`helm/rayservice/applications/prostate-classifier-1.yaml`,
`workflow-definitions/prostate-heatmap.yaml`, and `eads/prostate-heatmap.json`.
Its generated EAD demonstrates deriving the provider contract from one source,
but its singular top-level `model` link and generator are too narrow for the
general workflow design. It does not yet register or run the proposed shared
workflow.

## Execution flow

```text
Synchronous:
RationAI SDK run ─> workflow application ─> workflow ─> model inference(s) ─> result

Scheduled:
RationAI SDK/JES schedule ─> execution scheduler ─> workflow application ─> workflow
                         ^                       │
                         │                       v
                  provider finish <──── durable workflow result
```

For a new EMPAIA schedule request, the provider adapter fetches and validates
the EMPAIA job configuration before model-service commits the execution and returns
`202`. A repeated request for an already admitted execution returns it without
another configuration fetch. Two concurrent first schedule requests may
both perform the read-only fetch, while atomic admission still creates one
execution.
The execution scheduler sees execution state and resource demand, not EAD
fields or EMPAIA output objects. It invokes the workflow application, records
the result, then invokes provider finish. It persists the completed result or
a reference to it before provider finish so a restart can resume that work
without rerunning inference. It marks the execution successful only after the
required provider finish succeeds.
This is ordinary control flow, not a Ray Serve completion callback or a
separate pod. The first file-completion rule is defined in
[`execution-interfaces.md`](execution-interfaces.md). The execution scheduler
invokes the provider adapter for EMPAIA publication.

## Implementation focus after the architecture grill

The architecture grill is complete at this level. The remaining work is
interface design, implementation, or empirical validation. Do not turn every
failure race into another stakeholder question. Reopen the architecture only
if that work contradicts an agreed contract.

### Work that blocks the generic execution slice

1. Implement the invocation, result, status, error, lease, and provider
   contracts in [`execution-interfaces.md`](execution-interfaces.md).
2. Implement the PostgreSQL transactions and test competing claims, fencing,
   deadlines, cancellation, and the per-workflow limit against PostgreSQL.
3. Implement the HTTP workflow-dispatch adapter. The local prototype settled
   the first adapter choice; missing GPU capacity and Serve-controller loss
   still need a cluster test before production rollout.
4. Pick ordinary initial values for leases, backoff, and attempts, then make
   them configurable. These are implementation defaults, not new architecture
   questions.
5. Preserve the current trusted shared-path behavior for the first slice.
   Path translation, allowed roots, overwrite policy, and collision hardening
   remain deferred until the interface is exposed outside a trusted
   deployment.

### Work that blocks the EMPAIA slice

1. Define secure storage and in-process use of the original EMPAIA job token.
2. Test repeated EMPAIA output and finalize calls before enabling publication
   retries. Map model-service's coarse statuses into the JES wait contract.
3. Implement the deferred JES capability selection change. EAD registration
   and reconciliation do not block execution.

### Deferred work

- Record and artifact retention, cleanup, and quotas.
- Workflow-actor and model-actor placement or tile-transfer optimization.
- WSI level selection versus resampling to model target MPP, pending
  model-supervisor input.
- EAD registration and reconciliation.
- Authentication and per-user ownership for standalone scheduled execution.
- Exact cancellation races and stronger duplicate-processing suppression.
- Concrete public HTTP paths. The interface and domain names are settled in
  `execution-interfaces.md` and `CONTEXT.md`.

The earlier EMPAIA-only POC was a working target, not a requirement that
overrides standalone use.

### Settled module map

These names are the implementation vocabulary.

- A workflow application presents one small interface: run a validated
  invocation and return a normalized result. Its implementation hides WSI
  processing and any sequence of model inferences.
- The execution scheduler presents one interface with `schedule`, `get`, and
  `cancel` operations. Its implementation hides claims, retries, Ray
  dispatch, durable result commit, provider finish, and recovery.
- Ray dispatch is a remote seam owned by model-service. The production adapter
  calls Ray Serve; tests use an in-memory adapter with the same invocation and
  result contract.
- The execution repository is an internal seam of the execution scheduler
  module. PostgreSQL implements it in production. Tests should exercise the
  state machine against PostgreSQL where transaction behavior matters rather
  than growing a general database framework.
- Provider integration is a real seam because standalone and EMPAIA behavior
  differ. Each adapter prepares provider-specific context and finishes the
  durable workflow outcome. Workflow applications do not publish provider
  outputs.
- The workflow catalog loads repository-owned contracts and checks that every
  declared workflow identifier has a corresponding Ray Serve application. It
  does not describe the workflow's model graph.

## Current code versus the proposal

- `models/binary_classifier.py` accepts compressed, model-ready tiles. It does
  not read a WSI.
- `builders/heatmap_builder.py` currently writes a BigTIFF to a caller-supplied
  server-visible path and returns that path after a synchronous request. The
  RationAI SDK's `slide.heatmap` wraps this call. The shared-filesystem requirement
  preserves this broad shape. Exact path safety remains deferred; scheduled
  results use the normalized output contract in `execution-interfaces.md`.
- The older `../rationai-empaia-jobs` score scenario computes a float and
  posts it directly, while its heatmap scenario uploads pixelmap tiles during
  processing. The proposed independent publication step needs a retained
  result for either shape; it cannot reuse that streaming implementation
  unchanged. EAD outputs distinguish `float` and `continuous_pixelmap`.
- The EMPAIA job-token schema describes a JWT with an `exp` claim. The 24-hour
  setting in the infrastructure notes is for cleanup of completed JES pods,
  not an established token lifetime.
- `prototypes/empaia-control-plane/` and `docs/architecture/empaia-control-plane.md` are the
  disposable first iteration. They are EMPAIA-specific, use SQLite, and stop
  at `READY`. They do not implement the generic workflow execution proposed
  here.
- Names in that draft are also superseded. `ExecutionService` becomes the
  execution scheduler interface. `ExecutionSpec` splits into a workflow
  invocation and protected provider context. Its detailed `ExecutionState`
  remains internal and projects to the five-value execution status.
- The first draft's class split is not disposable as a whole. Its store owns
  atomic transitions and attempt fencing; its EMPAIA resolver, AppService
  client, storage-locator resolver, and reconciler each hide substantial
  behavior. Keep those responsibilities as internal modules, but compose them
  behind the provider-neutral execution-scheduler interface.
- The draft has no production path from `READY` through Ray dispatch, durable
  result recording, provider finish, and completion. Its domain objects mix
  generic scheduling fields with EMPAIA identities, and its HTTP interface
  exposes the internal state enum. Those parts conflict with the agreed
  generic execution record and coarse public status.
- The draft resolves EMPAIA configuration after committing `ACCEPTED`, starts
  its runtime deadline only on the first execution claim, and uses fixed retry
  delays. The agreed flow instead validates configuration before admission,
  applies the deadline to the full scheduled lifetime, and uses capped
  exponential backoff.
- The draft EAD generator requires every workflow definition to name one model
  deployment. That is an example-specific shortcut, not the workflow contract:
  programmatic workflows may compose several model inferences, and EAD
  generation should depend on the provider-facing input/output definition.

## Superseded assumptions in earlier notes

Notes `../notes/16-empaia-model-service-component-walkthrough.md` and
`../notes/17-new-architecture-overview.md` remain research history. Do not
carry these assumptions forward without reopening them:

- An EMPAIA-only durable execution store is no longer the target. Scheduled RationAI SDK
  workflows need the same provider-neutral scheduler.
- EMPAIA configuration resolution no longer follows `202`. The provider
  adapter fetches and validates EMPAIA job configuration before committing the
  execution, so the durable record has the resolved execution values.
- “No JES core changes” is no longer fixed. A small backend-capability change
  is acceptable, although deferred.
- Model-service should not implement EMPAIA storage mappings or require JES to
  send a WSI path. The EMPAIA client owns that access abstraction.
- A separate MDS confirmation poll after successful finalize calls is not
  required.
- A GPU worker doing all WSI reading and workflow orchestration in process is
  not the assumed topology. CPU workflow and GPU model actors are expected;
  their placement and data transfer need measurement.
- EAD registration through CI/CD is no longer preferred. The exact deployed
  service reconciliation mechanism is deferred.

The standalone Ray Serve architecture pages describe the existing tile-inference
routes. They do not yet specify the proposed workflow execution system.
