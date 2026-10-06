# Model execution

This context names the work model-service runs and how callers track it,
independently of the system that supplies the data.

## Work

**Model inference**:
A call to one deployed model with model-ready input that returns a prediction.
It does not orchestrate whole-slide processing.
_Avoid_: Model operation, low-level model call

**Workflow**:
A user-meaningful computation over a WSI or ROI that may use one or more model
inferences and produces declared outputs.
_Avoid_: Builder, scenario, job

**Workflow definition**:
A repository-owned declaration of a workflow's identity and input and output
contract. It may also contain metadata for an EMPAIA application definition.
_Avoid_: Workflow configuration, model configuration

**Workflow invocation**:
One request to run a workflow with concrete inputs and resolved configuration.
It may be synchronous or belong to a scheduled execution.

**Workflow result**:
The named outputs produced by a workflow invocation. Each output is either an
inline value or a shared artifact.

**Shared artifact**:
An explicit workflow-result reference to a completed file on shared storage.
It is distinct from an inline string value.
_Avoid_: Bare output path

## Invocation and execution

**Synchronous invocation**:
A workflow invocation for which the caller keeps the request open and receives
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

**Processing attempt**:
One try to produce the workflow result of a scheduled execution. A scheduled
execution may have several processing attempts but only one accepted result.
_Avoid_: Job attempt

**Execution status**:
The public state of a scheduled execution: `queued`, `running`, `succeeded`,
`failed`, or `cancelled`.
_Avoid_: Job status, phase

**Execution deadline**:
The latest time by which a scheduled execution may complete its workflow and
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
The integration context that supplies workflow inputs and consumes the workflow
result or terminal state. The current providers are standalone shared storage
and EMPAIA.

**Provider adapter**:
The translation between provider-specific scheduling and finish behavior and
the common scheduled-execution model.

**Standalone provider**:
The provider used by RationAI SDK callers that exchange workflow inputs and
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
The successful EMPAIA provider finish that uploads the workflow result and
finalizes the EMPAIA job.

**Workflow input access**:
The provider-neutral view through which a workflow reads its named inputs.
Provider transport details do not change workflow logic.

**Shared-storage path**:
A path that both a standalone caller and the workflow runtime resolve to the
same file on shared storage.
_Avoid_: Workflow path, local path

## Configuration and deployment

**Resolved workflow configuration**:
The concrete workflow values accepted for one invocation after validation
against the workflow definition.
_Avoid_: Execution spec, model parameters

**Workflow application**:
The deployed runtime that hosts a workflow under its stable workflow
identifier.
_Avoid_: Workflow endpoint, builder application

**Model deployment**:
The deployed runtime that loads a model and owns the settings needed for its
model inferences.
_Avoid_: Model workflow

**Model target MPP**:
The physical pixel resolution a model expects based on its training data.
It is a property of the model deployment, not a provider setting.

**EMPAIA application definition (EAD)**:
The EMPAIA declaration corresponding to a workflow definition and its optional
EMPAIA metadata.
_Avoid_: EAD configuration, model configuration

## Named clients and JES

**RationAI SDK**:
The public client library for model inference and workflow invocation through
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
