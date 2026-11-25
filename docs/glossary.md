# Model execution

This context names the work model-service runs and how callers track it,
independently of the system that supplies the data.

## Processes

**Control plane**:
The model-service process outside Ray that manages work: it serves the
catalog, admits and resolves runs, keeps scheduled executions in PostgreSQL
and integrates the providers. It never handles tiles, so raw inference skips
it.
_Avoid_: API (for the process), gateway

**Data plane**:
The Ray Serve applications that do the compute: one per model, and the
pipeline runner. Callers reach it directly only for raw inference; pipelines
come through the control plane.
_Avoid_: Execution plane, workers

## Building blocks

**Component**:
A small reusable piece of code that catalog files name by its name. Its kind
decides where it may be used: a **loader** fetches weights, a **runtime** runs
them, a **step** transforms data item by item, a **source** produces a
pipeline's items from its inputs, a **sink** collects results, an **input
converter** turns an EMPAIA input into a run input, and a **publisher** carries
an output into EMPAIA.
_Avoid_: Unit, plug-in, module

**Sink**:
The component that collects a pipeline's results into one output file of a
data format. A pipeline may have several, written all or nothing.
_Avoid_: Writer, output builder

**Data format**:
A named, versioned file shape that sinks write and readers open, identified by
a media type with a version. Readers refuse a version they don't know.
_Avoid_: Output type, schema (for the file layout)

**Catalog**:
The model and pipeline files of one deployment, checked together, with
the environment's patches applied.
_Avoid_: Model catalog, config

**Environment patch**:
A catalog file holding only what differs on one cluster, merged over the
base file of the same name.
_Avoid_: Override file, copy

## Work

**Model**:
One trained model deployed behind the generic host, assembled from a loader,
a runtime and optional steps, and declaring the tiles it takes and what it
returns: a file in `catalog/models/`, whose name is its model ID. Its trained
parameters are its weights.
_Avoid_: Served model, model deployment, inference implementation, model class

**Model output**:
What a model declares it returns per tile: scores (a score map of
some channels) or an embedding (one vector per summary). It replaces the
inference contract.
_Avoid_: Inference contract, model type

**Model digest**:
A fingerprint of everything that decides what a model computes. A run
whose model's digest differs from the one it was admitted with fails rather
than produce results from another model.

**Model inference**:
A call to one model with one model-ready tile that returns its output.
It does not orchestrate whole-slide processing.
_Avoid_: Model operation, low-level model call

**Pipeline**:
One slide-in, results-out job described in one catalog file: a source, steps
that may call models, and sinks, in that order. A file in
`catalog/pipelines/`, whose name is its public ID. A pipeline runs models.
_Avoid_: Analysis, builder, workflow, scenario

**Setting**:
An option of a pipeline's components that a caller may change for one run,
such as the stride between tiles. EMPAIA offers settings as job parameters.
_Avoid_: Builder configuration, parameter (outside EMPAIA)

**Provider settings**:
How a pipeline is offered beyond the public API, under its file's
`providers:` key. A pipeline with `providers.empaia` is one EMPAIA app; the
standalone provider needs no settings and is always available.
_Avoid_: Exposure, registration, publishing (for the app itself)

**Run**:
One request to execute a pipeline with concrete inputs, settings and an
output path. It executes the pipeline as it was resolved when the run was
admitted.
_Avoid_: Job, inside model-service

**Result**:
The named outputs of a run. Each output is an inline value or a shared
artifact.
_Avoid_: Builder result

**Shared artifact**:
A result reference to a completed file on shared storage, distinct from an
inline string value.
_Avoid_: Bare output path

## Runs and executions

**Synchronous run**:
A run for which the caller keeps the request open and receives the result
directly. It has no durable execution record.
_Avoid_: Real-time request, direct request

**Schedule request**:
A request to create or retrieve a scheduled execution. Repeating an equivalent
request refers to the same scheduled execution.
_Avoid_: Batch request, job submission

**Submission identity**:
The provider-scoped identity used to recognize repeated schedule requests.
Standalone schedule requests name it with the `Idempotency-Key` header;
EMPAIA schedule requests use the EMPAIA job identity.

**Scheduled execution**:
The durable model-service lifecycle created by one schedule request, including
any repeated processing attempts and provider finish.
_Avoid_: Batch job, model-service job

**Processing attempt**:
One try to produce the result of a scheduled execution. A scheduled execution
may have several processing attempts but only one accepted result.
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
_Avoid_: Demand controller, control plane (the scheduler is one part of it), job scheduler

**Model capacity**:
How many scheduled runs may use one model at once. A run is claimed
only when every model it calls has room; a pipeline without models
has capacity of its own.

**Synchronous headroom**:
The best-effort share of model capacity left available to synchronous runs
while scheduled executions run.

## Provider integration

**Provider**:
The integration context that supplies a run's inputs and consumes its result
or terminal state. The current providers are standalone shared storage and
EMPAIA.

**Provider adapter**:
The translation between provider-specific scheduling and finish behavior and
the common scheduled-execution model.

**Standalone provider**:
The provider used by RationAI SDK callers that exchange inputs and artifacts
through shared storage without an external job lifecycle.
_Avoid_: Direct provider, filesystem mode

**EMPAIA provider**:
The provider that relates a scheduled execution to an EMPAIA job and its input
and publication conventions.

**EMPAIA app**:
What EMPAIA lists and starts jobs of. Each pipeline offered through EMPAIA is one
app.
_Avoid_: App, when meaning a Ray Serve application

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
The successful EMPAIA provider finish that uploads the result and finalizes
the EMPAIA job.

**EMPAIA application definition (EAD)**:
The EMPAIA declaration of one app, generated from the pipeline's EMPAIA
provider settings and the fragments its converters and publishers contribute.
_Avoid_: EAD configuration, model configuration

**Input access**:
The provider-neutral view through which a run reads its named inputs.
Provider transport details do not change the pipeline.

**Shared-storage path**:
A path that both a standalone caller and the pipeline runner resolve to the
same file on shared storage.
_Avoid_: Local path

## Slides

**Model target MPP**:
The physical pixel resolution a model expects, from its training data.
Pipelines read the slide level closest to it.

**Region of interest**:
One or more named areas limiting a pipeline to part of a slide, sent as a
[`regions`](reference/formats.md#regions) file: GeoJSON polygons in level-0
pixels, each with an ID. A pipeline requires it, accepts it, or ignores it.
Per-region sinks such as `region_fraction` report one number per ID.
_Avoid_: ROI mask

**Tissue mask**:
A mask separating tissue from background, so a pipeline can skip empty
glass. Pipelines compute their own from colour saturation until QC publishes
one.
_Avoid_: Coverage mask

## Quality control

**QC (quality control)**:
RationAI's quality-control service: a Ray Serve application that runs QC checks
on a whole-slide image read from a path and writes their masks next to it.
_Avoid_: Preprocessing, when meaning QC

**QC check**:
One detector of the QC service: residual artifacts, folding or blur, optionally
after white-balance correction.

**Artifact mask**:
A downsampled QC output marking where a QC check found artifacts; any non-zero
value means an artifact. The residual check's "coverage mask" is one.
_Avoid_: Coverage mask, when meaning tissue; tissue mask

## Named clients and JES

**RationAI SDK**:
The public client library for models, pipelines and executions in
model-service, with readers for every data format.
_Avoid_: SDK, when the intended library is ambiguous

**EMPAIA client**:
The SDK's `rationai.empaia.Client`, used inside the EMPAIA provider
integration to access job inputs and publish outcomes.
_Avoid_: EMPAIA SDK

**Execution backend**:
A JES integration that can run a class of registered EMPAIA applications, such
as a container runtime or model-service.
_Avoid_: Job executor

**Backend capability check**:
A side-effect-free decision by an execution backend about whether it supports
a registered EMPAIA application.
