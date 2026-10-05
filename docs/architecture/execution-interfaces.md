# Workflow execution interfaces

Status: historical interface sketches, 2026-09-16. Use
[the documentation map](../README.md) for the primary specification and current
package contracts. HTTP routes now exist; these sketches are not their schema.
Persistence now also exposes readiness, independent maintenance, and fenced
provider-mutation marking. In particular, a lost provider-finish lease after a
mutation was marked must not automatically replay publication. The original
sketches below predate that safety rule.

This page turns the decisions in
[`workflow-execution-proposal.md`](workflow-execution-proposal.md) into small
module interfaces. The Python sketches describe contracts. They are not a
requirement to copy these declarations into one file.

## Design summary

Ordinary callers see two modules:

- A workflow application has one operation, `run`.
- The execution scheduler has three operations, `schedule`, `get`, and `cancel`.

The execution-scheduler implementation owns three internal seams:

- Workflow dispatch has one operation, `run`.
- Persistence has seven semantic transaction operations.
- A provider adapter has three operations, `identify`, `prepare`, and
  `finish`.

This keeps Ray queues, leases, retries, provider finish, and PostgreSQL
states out of caller code. It also keeps EMPAIA types out of workflows.

## Ray prototype review

The prototype in
`prototypes/empaia-control-plane/control_plane/prototypes/ray_dispatch_logic_prototype.html`
supports HTTP as the first production workflow-dispatch adapter.

- A request woke a deployment with zero replicas. Scheduled admission does
  not need an active replica check.
- Confirmed Ray queue rejection is backpressure, not a workflow attempt.
- Waiting on a deployment handle with a timeout did not cancel its work.
- Killing an HTTP client caused Serve to cancel the handler. Killing a handle
  client left the handler running.

HTTP therefore reduces duplicate GPU work after an execution-scheduler
process dies. It does not prevent it. Disconnect detection and Ray cancellation
are best effort, blocking workflow code may not observe cancellation, and a
lost response can leave the outcome unknown. Durable leases and fencing remain
the authority.

The prototype simplifies four production concerns:

1. Its outstanding-request counter is in memory. PostgreSQL must enforce the
   per-workflow limit across execution-scheduler replicas.
2. Its claim has no explicit fencing token. Every production settlement must
   present the current token.
3. It treats an HTTP disconnect as immediate cancellation. The production
   adapter can only request cancellation and then rely on fencing.
4. It marks a stored result successful. A provider such as EMPAIA still needs
   successful provider finish before the execution becomes `succeeded`.

Use one outstanding scheduled workflow request per workflow initially. Keep
the value configurable and do not derive it from Ray autoscaling settings.

## Shared values

```python
type JsonValue = (
    None | bool | int | float | str
    | list[JsonValue]
    | dict[str, JsonValue]
)


class ProtectedJson:
    """Opaque secret-bearing JSON protected by persistence and serialization."""


@dataclass(frozen=True, slots=True)
class InlineInput:
    value: JsonValue


@dataclass(frozen=True, slots=True)
class SharedPathInput:
    path: str


@dataclass(frozen=True, slots=True)
class ProviderInput:
    name: str


type WorkflowInput = InlineInput | SharedPathInput | ProviderInput


@dataclass(frozen=True, slots=True)
class InputAccessContext:
    kind: str
    state: dict[str, JsonValue]
    protected: ProtectedJson | None


@dataclass(frozen=True, slots=True)
class WorkflowInvocation:
    inputs: dict[str, WorkflowInput]
    resolved_configuration: dict[str, JsonValue]
    input_access: InputAccessContext
    deadline_at: datetime


@dataclass(frozen=True, slots=True)
class InlineOutput:
    value: JsonValue


@dataclass(frozen=True, slots=True)
class SharedArtifact:
    path: str
    media_type: str | None = None


type WorkflowOutput = InlineOutput | SharedArtifact


@dataclass(frozen=True, slots=True)
class WorkflowResult:
    outputs: dict[str, WorkflowOutput]


@dataclass(frozen=True, slots=True)
class ExecutionError:
    code: str
    message: str
```

All variants have explicit tags in their serialized representation. An
ordinary string is never inferred to be a path. `ProtectedJson` is a domain
type whose values must not appear in logs, fingerprints, public views, or raw
database columns. The PostgreSQL implementation must encrypt it at rest. The
encryption mechanism is part of the EMPAIA slice and does not change these
interfaces.

`InputAccessContext` is transport data. Programmatic workflow code does not inspect
it. The workflow-application implementation selects its input-access adapter
and gives the workflow provider-neutral readers. This is where an EMPAIA
client is constructed inside the Ray process. The execution scheduler never
converts an EMPAIA WSI into a filesystem path.

## Workflow application module

```python
class WorkflowApplication(Protocol):
    async def run(
        self,
        invocation: WorkflowInvocation,
    ) -> WorkflowResult: ...
```

This one-operation interface is shared by synchronous calls and scheduled Ray
dispatch. The implementation hides:

- validation against the repository-owned workflow definition;
- construction of standalone or EMPAIA input access;
- WSI reading, tiling, and model-inference ordering;
- progress and cancellation checks;
- output normalization and contract validation;
- artifact completion checks.

Each workflow may parse the generic invocation into its own typed values. It
may call zero, one, or several model deployments. It does not read a scheduled
execution record or publish a result to EMPAIA.

The interface has these failure classes:

```python
class InvalidWorkflowInvocation(Exception): ...
class WorkflowContractViolation(Exception): ...


@dataclass(frozen=True, slots=True)
class WorkflowFailure(Exception):
    code: str
    safe_message: str
    retryable: bool
```

Invalid input is permanent. A contract violation means the deployed workflow
does not match its definition. An unexpected exception becomes a generic safe
infrastructure error at the HTTP ingress; its raw text stays in protected
logs.

For the first shared-filesystem slice, a file-producing workflow writes a
sibling temporary file, closes it, and atomically replaces the final path
before returning. The workflow application then verifies that the returned
path is a regular file. It does not require a non-empty file, checksum, or
artifact version. Returning `SharedArtifact` is the completion signal.

## Execution scheduler module

```python
class ExecutionStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class ProviderScheduleRequest:
    kind: str
    body: object


@dataclass(frozen=True, slots=True)
class ScheduleRequest:
    workflow_id: str
    provider: ProviderScheduleRequest
    requested_timeout: timedelta | None


@dataclass(frozen=True, slots=True)
class ExecutionView:
    execution_id: UUID
    workflow_id: str
    status: ExecutionStatus
    created_at: datetime
    updated_at: datetime
    deadline_at: datetime
    result: WorkflowResult | None
    error: ExecutionError | None


@dataclass(frozen=True, slots=True)
class ScheduleReceipt:
    execution: ExecutionView
    created: bool


class ExecutionScheduler(Protocol):
    async def schedule(
        self,
        request: ScheduleRequest,
    ) -> ScheduleReceipt: ...

    async def get(self, execution_id: UUID) -> ExecutionView: ...

    async def cancel(self, execution_id: UUID) -> ExecutionView: ...
```

Each HTTP ingress owns a strongly typed request model.
`ProviderScheduleRequest` is the internal wrapper used after the ingress
selects a provider adapter; it is not a proposal for one public catch-all JSON
body.

The implementation hides provider preparation, duplicate detection, durable
admission, claims, dispatch, heartbeats, retries, result storage, provider
finish work, recovery, and public status projection.

`schedule` follows this order:

1. Select the provider adapter and derive a provider-scoped identity and
   fingerprint without remote calls.
2. Return an equivalent admitted execution, or reject conflicting reuse.
3. Reject an unknown workflow.
4. Ask the provider adapter to prepare and validate a new execution.
5. Commit the complete prepared execution and its absolute deadline.
6. Return only after that transaction commits.

The admission transaction repeats the identity check. Two concurrent first
schedule requests may both perform read-only preparation, but only one
execution is created. This avoids a separate reservation lifecycle. It can be
tightened later if duplicate preparation becomes costly or gains side effects.

`get` exposes only the five public statuses. A processing retry waiting to run
is `queued`. Processing, provider finish, and provider-finish retry are
`running`. A result is public only after the execution succeeds.

`cancel` is idempotent. It atomically marks a live execution `cancelled` and
fences its lease, then cancels a locally owned dispatch task. Ray cancellation
is best effort. A stale completion cannot change the terminal record.

Caller-facing errors are `UnknownWorkflow`, `InvalidScheduleRequest`,
`ScheduleConflict`, `SchedulingUnavailable`, and `ExecutionNotFound`.
Failures after admission become durable `ExecutionError` values.

## Workflow-dispatch seam

Ray Serve is remote but owned. Production and in-memory adapters make this a
real seam.

```python
@dataclass(frozen=True, slots=True)
class DispatchRequest:
    workflow_id: str
    processing_attempt_id: UUID
    invocation: WorkflowInvocation


class WorkflowDispatch(Protocol):
    async def run(
        self,
        request: DispatchRequest,
    ) -> WorkflowResult: ...
```

The first production adapter uses HTTP. It resolves the workflow through the
catalog, sends one request to the Ray Serve route, applies the invocation's
absolute deadline, validates the response, and returns a normalized result.
Cancelling `run` must close the in-flight HTTP exchange so Serve can observe a
disconnect. There is no separate `cancel` operation because a replacement
execution scheduler cannot address an old HTTP request.

The adapter exposes semantic failures:

```python
class DispatchRejected(Exception):
    retry_after: timedelta | None


class DispatchOutcomeUnknown(Exception): ...


@dataclass(frozen=True, slots=True)
class RemoteWorkflowFailure(Exception):
    error: ExecutionError
    retryable: bool
```

`DispatchRejected` means the adapter can prove Ray did not accept the work. It
requeues without consuming a processing attempt. Do not map every HTTP `503`
to this class. A generic proxy or deployment failure can also return `503`;
without a recognized Ray backpressure response, the outcome is unknown.

A connection loss after sending the request is `DispatchOutcomeUnknown` and
consumes an attempt because the workflow may have run. The interface makes no
exactly-once claim. `processing_attempt_id` is for tracing and durable fencing,
not remote deduplication.

The in-memory adapter can return a result, reject capacity, block until
cancelled, or simulate an unknown outcome. It does not need to imitate HTTP.

## Persistence seam

The persistence seam names atomic scheduler transactions rather than database
CRUD or every internal state transition.

```python
@dataclass(frozen=True, slots=True)
class SubmissionIdentity:
    provider: str
    key: str
    fingerprint: str


@dataclass(frozen=True, slots=True)
class WorkLease:
    execution_id: UUID
    token: UUID
    expires_at: datetime
    work: RunWorkflow | FinishProvider


@dataclass(frozen=True, slots=True)
class RunWorkflow:
    workflow_id: str
    invocation: WorkflowInvocation


@dataclass(frozen=True, slots=True)
class FinishProvider:
    provider: str
    context: ProtectedProviderContext
    outcome: ProviderOutcome


class LeaseDirective(StrEnum):
    CONTINUE = "continue"
    CANCEL = "cancel"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    LEASE_LOST = "lease_lost"


type WorkOutcome = (
    WorkflowCompleted
    | ProviderFinished
    | RetryLater
    | WorkFailed
)


class ExecutionRepository(Protocol):
    async def find_submission(
        self,
        identity: SubmissionIdentity,
    ) -> InternalExecution | None: ...

    async def admit(self, new: NewExecution) -> Admission: ...

    async def get(self, execution_id: UUID) -> InternalExecution: ...

    async def request_cancel(
        self,
        execution_id: UUID,
    ) -> InternalExecution: ...

    async def claim_due(
        self,
        owner: str,
        lease_for: timedelta,
        policy: ClaimPolicy,
    ) -> Sequence[WorkLease]: ...

    async def renew(
        self,
        token: UUID,
        lease_for: timedelta,
    ) -> LeaseDirective: ...

    async def settle(
        self,
        token: UUID,
        outcome: WorkOutcome,
    ) -> InternalExecution: ...
```

These seven operations are the internal interface. Replacing them with
`read`, `claim`, and `apply` would reduce the method count while exposing a
larger command language. The semantic transactions are easier to audit and
test.

The PostgreSQL adapter owns these guarantees:

- `admit` atomically enforces provider identity and fingerprint uniqueness.
- `claim_due` recovers expired leases, observes retry times and deadlines, and
  enforces the per-workflow processing limit across execution-scheduler
  replicas.
- A workflow lease and a provider-finish lease are distinct work kinds. Both
  project to public `running` while active.
- `settle(WorkflowCompleted)` stores the result and creates provider-finish
  work in the same transaction.
- `settle(ProviderFinished)` marks a successful execution `succeeded`.
- `settle(WorkFailed)` marks the execution `failed` and may create
  best-effort provider-finish work for that terminal error. Failure-report
  exhaustion does not change the local terminal state.
- `renew` and `settle` reject a stale fencing token.
- Cancellation and deadline expiry fence the active token before returning.
- An expired workflow lease consumes a processing attempt and requeues work if
  policy permits. An expired provider-finish lease retains the result and
  retries only provider finish.
- Confirmed dispatch rejection uses `RetryLater` without consuming an attempt.
- Backoff, maximum processing attempts, and maximum provider-finish attempts
  are evaluated in the transaction.
- Database time determines lease and deadline comparisons.
- Protected provider state never appears in public projections.

PostgreSQL should use row locking such as `FOR UPDATE SKIP LOCKED` for claims.
That is an implementation choice behind the interface, not a caller
obligation.

Use an in-memory adapter for scheduler orchestration tests. Run identity races,
competing claims, per-workflow limits, lease recovery, fencing, cancellation,
and deadlines against PostgreSQL. The in-memory adapter is not evidence that
transaction behavior is correct.

Processing attempt accounting starts when the outcome proves Ray accepted the
request or leaves acceptance uncertain. A confirmed pre-acceptance rejection
does not consume an attempt. If the execution scheduler dies with an active
workflow lease, recovery consumes that attempt because the remote outcome is
unknown.

## Provider-adapter seam

Standalone filesystem and EMPAIA behavior already differ, so this seam is
real.

```python
@dataclass(frozen=True, slots=True)
class PreparedExecution:
    invocation: WorkflowInvocation
    context: ProtectedProviderContext
    deadline_cap: datetime | None


@dataclass(frozen=True, slots=True)
class ResultOutcome:
    result: WorkflowResult


@dataclass(frozen=True, slots=True)
class FailureOutcome:
    error: ExecutionError


@dataclass(frozen=True, slots=True)
class CancellationOutcome: ...


type ProviderOutcome = (
    ResultOutcome | FailureOutcome | CancellationOutcome
)


class ProviderAdapter(Protocol):
    def identify(
        self,
        request: ProviderScheduleRequest,
        workflow_id: str,
    ) -> SubmissionIdentity: ...

    async def prepare(
        self,
        request: ProviderScheduleRequest,
        workflow: WorkflowDefinition,
        requested_timeout: timedelta | None,
    ) -> PreparedExecution: ...

    async def finish(
        self,
        context: ProtectedProviderContext,
        outcome: ProviderOutcome,
    ) -> None: ...
```

`identify` is local and side-effect free. Its fingerprint excludes credentials.
`prepare` performs provider-specific resolution before admission and returns a
validated invocation, protected context, and optional deadline cap. `finish`
performs required success finish or a best-effort failure or cancellation
report.

The standalone adapter derives identity from the caller's idempotency key,
validates shared paths and configuration, builds filesystem access, and has a
no-op `finish`.

The EMPAIA adapter derives identity from the installation and EMPAIA job,
fetches configuration during `prepare`, builds EMPAIA access without
translating the WSI into a path, and caps the deadline at the original token
expiry. On
success, `finish` uploads the normalized result and finalizes the EMPAIA job. A
successful execution requires that call to return. Failure and cancellation
reports are best effort and cannot reverse the local terminal state.

Success finish work remains `running` until it succeeds or exhausts its retry
policy. Finish work for failure and cancellation runs after the local terminal
state and is discarded after its small retry limit or credential deadline.

The execution-scheduler implementation owns retry timing and limits. The
adapter classifies failures as permanent, retryable, or outcome unknown.
Retry an outcome-unknown EMPAIA publication only after its operation's
duplicate behavior has been verified.

Provider input access inside Ray is a separate internal seam because live
Python objects cannot cross HTTP. `InputAccessContext` connects the scheduler-side
provider adapter to the Ray-side input-access adapter. Workflow code sees
provider-neutral readers rather than this transport object.

## Durable state model

The public state is a projection of a smaller internal work model:

| Durable condition | Public status | Next work |
| --- | --- | --- |
| Workflow waiting for a lease | `queued` | `RunWorkflow` |
| Workflow lease active | `running` | Await dispatch |
| Result stored, provider finish due | `running` | `FinishProvider` |
| Provider finish lease active or retry waiting | `running` | Finish or retry |
| Provider success finished | `succeeded` | None |
| Terminal processing or provider-finish error | `failed` | Optional best-effort provider finish |
| Cancellation committed | `cancelled` | Optional best-effort provider finish |

The result-storage transaction is the dividing line between processing retry
and provider-finish retry. A finish retry never reruns the workflow.

## Designs compared

Three shapes were considered.

The minimal shape used a three-method persistence interface with typed
commands. It made the method list short, but callers still had to learn every
command and legal ordering. It also proposed schedule-request reservations to
avoid duplicate preparation during concurrent first requests. That was more
state than the current read-only preparation warrants.

The flexibility-first shape modeled workflow processing and provider finish
as leased actions. That put recovery and retry rules in one place and made
multi-output workflows and new providers cheap to add. Its open resource
drivers were broader than the two current adapters require.

The caller-first shape kept synchronous `run` and scheduled
`schedule`/`get`/`cancel` as the only ordinary entry points. It used semantic
repository transactions and one-operation dispatch. This gives the best depth
and locality for the current system.

The settled design combines the caller-first external interfaces with the
leased-work persistence model. It keeps provider requests strongly typed at
their ingress, avoids a generic plug-in framework, and does not add a
schedule-request reservation lifecycle yet.

## Effect on the exploratory control plane

Keep the draft's atomic-transition, fencing, safe-error, provider-client error
classification, and lost-worker recovery ideas. Replace its interfaces:

- The ten-state `ExecutionState` must not remain public.
- `ExecutionSpec` cannot assume one model, one mounted WSI, or one output.
- EMPAIA configuration resolution moves before durable admission.
- Result storage and provider finish become separate leased work.
- The production adapter uses PostgreSQL rather than SQLite.
- The store's per-transition methods collapse into the seven semantic
  transactions above.
- The resolver, reconciler, and store become implementation details behind
  `ExecutionScheduler` rather than caller-visible modules.

Public URI names, encryption for protected provider state, and the concrete
RationAI input-reader interface remain implementation tasks. They do not need
to widen the interfaces on this page.
