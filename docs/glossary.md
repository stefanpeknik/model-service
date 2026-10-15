# Model execution

This context names the work model-service runs and how callers track it,
independently of the system that supplies the data.

## Work

**Model inference**:
A call to one deployed model with model-ready input that returns a prediction.
It does not orchestrate whole-slide processing.
_Avoid_: Model operation, low-level model call

**Inference implementation**:
Code in `src/model_service/models/` that serves one kind of model-ready input and output. It can
serve any trained model that fits it.
_Avoid_: Model, when meaning the code rather than a deployment

**Model deployment**:
One trained model served through an inference implementation, identified by its
model ID and declared in one `deploy/helm/rayservice/models/` file.
_Avoid_: Model workflow

**Inference contract**:
The model-ready input and prediction structure a model deployment declares:
`spatial-scores` or `tile-embedding`. Builders accept one contract.
_Avoid_: Model type

**Model catalog**:
The facade's list of model deployments and their contracts, rendered from the
same files that deploy the models.

**Builder**:
Reusable whole-slide processing that calls one compatible model deployment and
writes declared outputs, for example `heatmap` or `tile-embeddings`.
_Avoid_: Workflow, pipeline, scenario

**Invocation**:
One request to run a builder with concrete inputs. The model is chosen per
request; processing settings are the builder's current defaults, or the
EMPAIA job's configuration.

**Builder result**:
The named outputs of an invocation. Each output is an inline value or a shared
artifact.

**Shared artifact**:
A result reference to a completed file on shared storage, distinct from an
inline string value.
_Avoid_: Bare output path

## Invocation and execution

**Synchronous invocation**:
An invocation for which the caller keeps the request open and receives
the result directly. It has no durable execution record.
_Avoid_: Real-time request, direct request

**Schedule request**:
A request to create or retrieve a scheduled execution. Repeating an equivalent
request refers to the same scheduled execution.
_Avoid_: Batch request, job submission

**Submission identity**:
The provider-scoped identity used to recognize repeated schedule requests.
Standalone schedule requests use an idempotency key; EMPAIA schedule requests
use the EMPAIA job identity.

**Scheduled execution**:
The durable model-service lifecycle created by one schedule request, including
any repeated processing attempts and provider finish.
_Avoid_: Batch job, model-service job

**Scheduled inference**:
A scheduled execution of one raw model request (one tile) instead of a
builder. It has no provider finish; its stored response is downloaded
separately and expires after a retention period.
_Avoid_: Batch inference, tile job

**Processing attempt**:
One try to produce the builder result of a scheduled execution. A scheduled
execution may have several processing attempts but only one accepted result.
_Avoid_: Job attempt

**Execution status**:
The public state of a scheduled execution: `queued`, `running`, `succeeded`,
`failed`, or `cancelled`.
_Avoid_: Job status, phase

**Execution deadline**:
The latest time by which a scheduled execution may complete processing and
required provider finish.
_Avoid_: Model timeout, token timeout

**Provider finish**:
Provider-specific work after processing has a durable result, failure, or
cancellation. It delivers a successful result or reports the terminal state.
_Avoid_: Provider delivery, generic publication

**Execution scheduler**:
The part of model-service that owns schedule requests and advances scheduled
executions through processing and provider finish.
_Avoid_: Demand controller, control plane, job scheduler

**Synchronous headroom**:
The best-effort share of model capacity left available to synchronous
invocations while scheduled executions run.

## Provider integration

**Provider**:
The integration context that supplies builder inputs and consumes the builder
result or terminal state. The current providers are standalone shared storage
and EMPAIA.

**Provider adapter**:
The translation between provider-specific scheduling and finish behavior and
the common scheduled-execution model.

**Standalone provider**:
The provider used by RationAI SDK callers that exchange builder inputs and
artifacts through shared storage without an external job lifecycle.
_Avoid_: Direct provider, filesystem mode

**EMPAIA provider**:
The provider that relates a scheduled execution to an EMPAIA job and its input
and publication conventions.

**EMPAIA job**:
The external unit of work owned by EMPAIA and JES. One EMPAIA job refers to one
model-service scheduled execution.
_Avoid_: Execution, model-service job

**EMPAIA job context**:
The identity, authorization, and resolved provider values that relate one
scheduled execution to its EMPAIA job.
_Avoid_: EMPAIA execution context

**EMPAIA credential deadline**:
The expiry of the original EMPAIA job credential, which bounds the related
scheduled execution.

**EMPAIA publication**:
The successful EMPAIA provider finish that uploads the builder result and
finalizes the EMPAIA job.

**Input access**:
The provider-neutral view through which a builder reads its named inputs.
Provider transport details do not change builder logic.

**Shared-storage path**:
A path that both a standalone caller and the builder runtime resolve to the
same file on shared storage.
_Avoid_: Local path

## Configuration and deployment

**Builder configuration**:
A builder's processing settings (stride, tile sizes, encoding) with defaults
in `src/model_service/builders/registry.py`. Standalone requests do not set them; EMPAIA jobs may.
_Avoid_: Execution spec, model parameters

**Model target MPP**:
The physical pixel resolution a model expects based on its training data.
It is a property of the model deployment, not a provider setting.

**EMPAIA application definition (EAD)**:
The EMPAIA declaration generated from an app file (builder, model and EMPAIA
metadata) under `deploy/empaia/apps/`.
_Avoid_: EAD configuration, model configuration

## Named clients and JES

**RationAI SDK**:
The public client library for model inference and builder runs through
model-service.
_Avoid_: SDK, when the intended library is ambiguous

**EMPAIA client**:
The client library used inside the EMPAIA provider integration to access job
inputs and publish outcomes.
_Avoid_: EMPAIA SDK

**Execution backend**:
A JES integration that can run a class of registered EMPAIA applications, such
as a container runtime or model-service.
_Avoid_: Job executor

**Backend capability check**:
A side-effect-free decision by an execution backend about whether it supports
a registered EMPAIA application.
