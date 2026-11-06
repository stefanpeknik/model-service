# Checkpoint 1 review, 2026-09-27

Historical review. Several scheduler findings below have since been resolved;
the EMPAIA findings were deferred. Use [the documentation map](../README.md)
and its scratchpad progress link for the current backlog and validation boundaries.

## Verdict

The design is worth keeping. I would not call this production-ready yet, and I
would not rewrite it to make it look cleaner. The remaining problems are mostly
failure handling in the production worker loop, not excessive abstraction.
Passing the demo and the existing component tests does not cover these cases.

No implementation fixes accompany this review.

## Scope and sources

- Baseline assumed for this review: `upstream/main`, commit
  `3d37a6ef4e2d90f5c75087a0037fa300c96c3b39`.
- Reviewed checkpoint: `HEAD` at `fadfb17`, the 34 commits after that baseline,
  and relevant uncommitted configuration and design documents.
- Main comparison command: `git diff upstream/main`. Untracked files were read
  separately because that command does not include them.
- Decision sources: [agreed decisions](../architecture/workflow-execution-proposal.md),
  [execution interfaces](../architecture/execution-interfaces.md), and
  [domain vocabulary](../../CONTEXT.md).
- Evidence sources: [operations record](../operations/durable-workflow-execution.md)
  and [demo post-mortem](../operations/validation-sdk-demo-postmortem-2026-09-25.md).

I did not find a separate original assignment or a full grilling transcript.
This is a comparison against the decisions recorded in this checkout, not a
claim that an external assignment is fully satisfied. The SDK repository and
JES implementation were not independently reviewed. No applicable on-disk
`AGENTS.md`, `AGENTS.override.md`, or `CLAUDE.md` was found in this checkout.

The design explicitly excludes the exploratory `control_plane/` implementation
as a source of truth. The production review therefore concentrates on
`execution_scheduler/`, `workflow_execution/`, `integrations/empaia/`, their
tests, and deployment wiring.

## Standards

### Keep these choices

- State decisions live in `transitions.py`; PostgreSQL owns transactions and
  locking. That is a useful separation, not abstraction for its own sake.
- Workflows produce results without publishing to EMPAIA. Provider finish uses
  a stored result, so a publication retry need not repeat inference.
- HTTP dispatch, provider access, and persistence have concrete reasons to be
  separate interfaces. Avoid collapsing them into a large scheduler class.
- Explicit output tags avoid guessing whether a string is a file path.
  Temporary files and atomic replacement give artifacts a clear completion
  point.
- Keeping ambiguous publication outcomes separate from safely retryable
  failures is correct. Do not remove that distinction while fixing reporting.

### Confirmed implementation defects

1. **P1. Lease loss can terminate the worker permanently.**
   In `worker.py:105-112`, an internally cancelled dispatch is re-raised when
   the record is still running with the same token. That is exactly the state
   returned when renewal discovers an expired lease before another claimant
   recovers it. `CancelledError` escapes `run()` because it is not an
   `Exception`. The ASGI process stays alive without a working scheduler loop.
   A fake-clock reproduction produced a cancelled worker task and a running
   execution record. Distinguish worker shutdown from cancellation of one claim.

2. **P1. Renewal exceptions silently stop heartbeats without stopping work.**
   An exception from `repository.renew()` ends only the renewal task.
   `run_once()` continues awaiting dispatch and later discards the renewal
   exception through `gather(..., return_exceptions=True)`. A reproduction with
   one transient repository error completed dispatch after lease expiry and
   accepted its result. Supervise the two tasks together and stop local work
   when ownership cannot be maintained. Review settlement's lease-expiry check
   as part of this change; it currently checks the token and execution deadline,
   but not lease expiry.

3. **P2. Busy workers stop advancing queued deadlines.**
   `run()` awaits a complete `run_once()`, including processing or publication,
   before claiming again. Deadline expiry and abandoned-lease recovery happen
   only in `claim_due()`. With one scheduler replica, a long workflow can leave
   another execution queued past its deadline and postpone cancellation reports
   until credentials expire. The same occurs when every replica is busy.
   A reproduction left a one-second request queued after ten seconds of
   database-clock time. Keep bounded dispatch concurrency, but run maintenance
   independently of an individual request's lifetime.

4. **P2. Capped backoff still overflows before applying its cap.**
   `RetryPolicy.backoff()` multiplies a `timedelta` before calling `min`.
   With the production five-second initial delay, `backoff(45)` raises
   `OverflowError` rather than returning 300 seconds. Confirmed 429 rejections
   are unbounded by the processing-attempt limit, so this is reachable after
   roughly three hours and fifteen minutes of sustained backpressure, within
   the six-hour default deadline. Saturate before multiplication.

5. **P2. Clipping a rectangle moves its right and bottom edges.**
   `image_io.py:109-112` clips the origin but keeps the original width and
   height. A rectangle at `x=-20` with width 10 is entirely outside the slide,
   yet the writer creates a ten-pixel region starting at zero. I confirmed the
   writer's selected dimensions with a fake image backend. Intersect the
   original bounds with the slide bounds, then derive width and height.

### Cleanup after correctness fixes

The code has many small files, but most own a real decision. I would resist a
file-count reduction exercise. More useful cleanup would be:

- Put the exploratory control-plane code behind an explicit archive or separate
  development path so contributors cannot confuse its auth and lifecycle with
  the production scheduler.
- Add safe execution IDs, work kinds, and attempt identifiers to worker error
  logs. Logging only the exception class makes production failures hard to
  diagnose; logging arbitrary exception text could expose provider credentials.
- Make the scheduler image reproducible. It currently installs open dependency
  ranges independently of `uv.lock`, so a rebuilt image need not match the
  tested dependency set.
- Review narrow HTTP-client protocols when touching their adapters. Some exist
  mainly to support fake clients; HTTPX mock transports may reduce duplicate
  transport contracts. This is optional, not a reason to redesign the modules.

## Spec

### Implemented in the intended shape

Synchronous and scheduled calls share workflow code. Admission is durable and
idempotent. PostgreSQL controls claims and fencing. Provider configuration is
resolved before admission, credentials have encrypted storage, and retained
results support provider finish without redispatch. Raw model inference still
has a public entry point.

The public origin uses the scheduler ASGI application as an HTTP proxy for
synchronous requests, rather than separate ingress routing directly to Serve.
It does not create durable work for those requests. Treat this as a documented
deployment choice with extra proxy load, not a broken execution model.

### Confirmed requirement gaps

6. **P1. Ray downtime removes the durable API from Kubernetes service routing.**
   `WorkflowDeploymentHealth.missing_workflows()` treats a failed route-table
   request as every workflow being missing. `/readyz` then fails, and the Helm
   readiness probe removes scheduler pods from ready service endpoints. New
   work, polling, and cancellation become unavailable during Ray loss even
   though PostgreSQL can still serve them. The agreed admission rule permits
   temporary Ray unavailability. Separate durable-API readiness from downstream
   workflow health, while retaining the missing-declaration check.

7. **P1. EMPAIA scheduling has no separate service authentication.**
   The agreed decisions defer standalone authentication but explicitly retain
   EMPAIA service authentication. `schedule_empaia()` has no authentication
   dependency or middleware, and the generic schedule endpoint also accepts
   `provider="empaia"`. The supplied job credential is sent to a caller-supplied
   AppService URL; it does not authenticate JES to model-service. Protect both
   admission paths before treating this as the production EMPAIA entry point.
   The old control-plane token settings do not protect this application.

8. **P2. Known publication failure never schedules the agreed failure report.**
   After exhausted, known provider-finish failures,
   `transitions.py:334-335` calls `_terminal_failure` without `report=True`.
   A reproduction with three safely retryable finish failures produced three
   success-publication attempts and no `ProviderFailure` call. Schedule bounded
   best-effort reporting while the original credential is valid. Preserve the
   deliberate suppression of follow-up mutations for unknown outcomes.

### Do not turn deferred work into checkpoint defects

Retention, quotas, standalone ownership, path-collision policy, exact
cancellation races, implementation-version pinning, EAD reconciliation, and
MPP resampling policy are explicitly deferred. They need release boundaries,
not speculative frameworks in this patch.

Real EMPAIA interoperability is still unproven by the mock AppService suite.
JES capability selection is deferred, and the recorded GPU-starvation recovery
and scaled-to-zero controller recovery checks remain incomplete. The demo's
small validation TIFF must not be described as proof of clinical heatmap
processing.

## Verification performed

- A focused run of scheduler, workflow, EMPAIA, HTTP dispatch, EAD, and native
  processor component tests passed: **59 tests**.
- Full local unit discovery reported 79 tests with nine import errors. Missing
  FastAPI, Starlette, Psycopg, and Cryptography prevented a full result. These
  are environment failures, not demonstrated application regressions.
- Used cached PyYAML for focused tests and cached Ruff 0.16.5. Ruff reported
  ten diagnostics over production packages, validation, and tests, including
  broad catches, redundant returns, and return annotations. These are separate
  from the functional findings and are not a claim about the pinned CI version.
- Deterministic probes confirmed worker-loop termination on lease loss,
  ignored renewal errors, delayed queued deadlines, omitted failure reporting,
  backoff overflow, downstream-health conflation, and incorrect rectangle
  clipping. These probes used test adapters and do not prove PostgreSQL races.
- No live cluster, GPU, real AppService, or PostgreSQL integration test was run.
- Production sources and existing tests were checked against their initial
  Git state after review. They are unchanged.

## Next session

Start with worker lifetime and renewal supervision. Add regression tests that
exercise `ExecutionWorker.run()`, not only individual `run_once()` calls. Test
queued expiry and terminal reporting while dispatch stays blocked. Then fix
backoff saturation and known-failure reporting.

Next, test admission, polling, and cancellation through the Kubernetes Service
while Ray is unreachable. Close the EMPAIA authentication bypass through the
generic route, and add negative-origin rectangle tests. Run the PostgreSQL
suite for the resulting lease changes before repeating deployed recovery tests.

Only then spend time on optional cleanup. The checkpoint needs stronger failure
behavior, not more architecture.
