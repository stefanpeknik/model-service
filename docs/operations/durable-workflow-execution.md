# Durable workflow execution

The execution scheduler accepts workflow requests, stores them in PostgreSQL,
and dispatches due work to Ray Serve. It is a long-lived deployment. It does
not create a Kubernetes pod for each execution.

The same public service also exposes synchronous workflow execution at
`POST /v1/workflows/{workflow_id}/run`. Scheduling uses the separate
`POST /v1/workflows/{workflow_id}/schedule` operation. Both paths dispatch the
same workflow invocation to the same Ray Serve application, so clients need
one model-service origin and synchronous processing cannot drift from scheduled
processing. A synchronous request is never converted into a durable execution;
cancelling the caller's request cancels its in-flight HTTP dispatch.

Legacy model-inference and slide operations also use this public origin. The
execution-scheduler ingress forwards their existing POST paths and binary
payloads to Ray Serve, so clients do not need an internal Ray Serve address.

## Initial operating values

| Setting | Default | Purpose |
| --- | ---: | --- |
| Lease duration | 60 seconds | Bounds recovery time after a scheduler stops |
| Renewal cadence | 20 seconds | One third of the lease duration |
| Worker poll interval | 0.5 seconds | Delay when no durable work is due |
| Initial retry backoff | 5 seconds | First retry delay |
| Maximum retry backoff | 300 seconds | Cap for exponential retry delay |
| Processing attempts | 3 | Whole-workflow attempts after uncertain outcomes |
| Provider-finish attempts | 3 | Required publication and best-effort report attempts |
| Default deadline | 6 hours | Queueing, processing, and provider finish combined |
| Maximum deadline | 24 hours | Service cap for caller-requested timeouts |
| Scheduled limit | 1 per workflow | Cross-replica active processing limit |

All values are Helm values and environment variables on the scheduler. The
interfaces between scheduling, persistence, dispatch, and providers do not
depend on the defaults.

The scheduler makes an at-least-once workflow-processing guarantee. A lost
HTTP response or expired processing lease consumes an attempt and can repeat
GPU work. PostgreSQL fencing prevents an older attempt from changing durable
state after replacement, cancellation, or deadline expiry.

Provider finish uses a different recovery rule. The scheduler durably marks
the first outbound provider mutation before sending it. A lost lease before
that mark can retry from the stored workflow result; a lost lease after the
mark ends with `provider_finish_outcome_unknown` and must be reconciled with
the provider. It cannot safely replay publication without a verified
idempotency guarantee. A failed local execution may therefore correspond to
an EMPAIA job that was already finalized.

## PostgreSQL and protected provider context

`DATABASE_URL` points at PostgreSQL. Each scheduler replica uses the same
database and can claim work independently. Claims use row locks and fencing
tokens; a transaction-scoped advisory lock enforces the per-workflow active
processing limit across replicas.

`PROVIDER_CONTEXT_KEY` is a URL-safe base64 encoding of exactly 32 random
bytes. Protected provider context uses AES-256-GCM with a fresh 96-bit nonce,
a versioned binary envelope, and fixed associated data. The ciphertext is
stored in a `BYTEA` column. Credentials are excluded from invocation JSON,
submission fingerprints, public execution views, and log messages.

Key rotation and re-encryption are not part of this delivery. Keep the active
key available for as long as any execution encrypted with it can remain in
the database.

## Ray Serve behavior

The workflow deployment uses Ray Serve 2.58 or newer and configures queue
rejection as HTTP 429 with `Retry-After`. The scheduler treats only that
signal as confirmed pre-acceptance backpressure. A generic 503, connection
loss, or malformed response has an unknown acceptance outcome and consumes a
processing attempt.

`/readyz` reads Ray Serve's route table. A declared workflow whose route is
missing makes readiness return 503 with
`workflow_application_unavailable`. Scheduling an undeclared identifier is a
separate 404 `unknown_workflow` response. Admission does not require a live
replica, so scale-from-zero and temporary capacity shortages remain durable
queueing cases.

## EMPAIA adapter

Set `executionScheduler.empaia.enabled`, provide a shared workspace claim,
and configure `executionScheduler.empaia.gatewayFactory` as a
`module:attribute` callable. The callable receives an `EmpaiaJobContext` with
the EMPAIA API URL, job ID, and original job credential. It returns the gateway
used by preparation, workflow input access, and provider finish. The gateway
module must be present in both the scheduler and Ray worker images.

The original credential is never refreshed. Its real expiry caps the complete
execution deadline. Workflow results remain durable while required output
publication is retried. An unknown publication outcome is not retried. Local
processing failure and cancellation are terminal before bounded best-effort
provider reporting begins.

## Failure validation

Validate a release by observing terminal state and fencing, not elapsed
startup time:

1. Schedule with the workflow at zero replicas and verify eventual success.
2. Remove available GPU capacity, verify the execution remains queued, then
   restore capacity and verify it proceeds.
3. Interrupt the Ray cluster and Serve controller separately. Verify recovery
   follows the processing-attempt policy and retains the execution record.
4. Stop a scheduler during HTTP dispatch. After lease expiry, verify a new
   scheduler receives a different fencing token and the old settlement is
   rejected.
5. Cancel an active HTTP request. Verify durable cancellation is immediate;
   remote cancellation remains best effort.
6. Repeat controller loss while the workflow is scaled to zero. Verify the
   execution proceeds after recovery or fails only through its configured
   deadline and retry policy.

The PostgreSQL test suite covers the deterministic persistence parts of these
scenarios with real concurrent transactions. Cluster-loss exercises belong in
the target KubeRay environment because local substitutes do not prove
controller, autoscaler, or GPU scheduling behavior.

## Validation record

On 2026-09-17, the local Ray Serve 2.58 integration observed all of the
following through the HTTP dispatch boundary:

- a deployment configured with zero initial replicas accepted work and scaled
  up to complete it;
- one active request plus one queued request caused the next request to receive
  the configured 429 backpressure response;
- cancelling the HTTP client task caused Ray Serve to report the client
  disconnect and cancel the replica request.

The real PostgreSQL suite also exercised competing admissions and claims,
database-clock deadlines, lease replacement, stale fencing tokens, provider
finish recovery, per-workflow limits, and encrypted provider context.

On 2026-09-24, the `model-service-validation` release in `peknik-ns` passed
public HTTP checks for a deadline expiring without Ray worker capacity and for
cancellation remaining terminal without capacity. The active RayCluster had
`replicas: 0`, `minReplicas: 0`, and `maxReplicas: 0` during those tests. The
release was then restored with `maxReplicas: 1`, and scheduled and synchronous
invocations produced the same result.

Changing only `validation.maxWorkerReplicas` through `helm upgrade` changed
the RayService spec but did not change its active RayCluster. For this
validation run, the release was uninstalled and reinstalled with the
zero-worker value after confirming there were no active executions. The
PostgreSQL deployment and shared PVC were outside the release and remained
in place. Do not use the value-only upgrade as evidence of zero worker
capacity; inspect the active RayCluster spec before running these tests.

The same release passed the guarded RayCluster-loss test twice. Each run
deleted the validation RayCluster during an in-flight EMPAIA direct-access
read. KubeRay activated a replacement cluster, and the durable execution
retried, succeeded, and finalized the EMPAIA job. The active RayService was
`Running` after both runs.

The guarded scheduler-replacement test also passed against this release. It
deleted the scheduler pod during an in-flight EMPAIA direct-access read; the
replacement scheduler recovered the durable execution and finalized the job.

The first guarded Serve-controller-loss test passed twice for the in-flight
EMPAIA execution, but its fault injection used `ray.kill(...,
no_restart=True)`. That overrides Serve's configured `max_restarts=-1`: a new
controller actor appeared, while the existing proxy retained a handle to the
dead actor. Ray 2.58 also defaults `ray.kill` to `no_restart=True`, so an
omitted argument has the same effect. After the deployment scaled to zero, a
later SDK execution queued at the proxy for about 599 seconds and failed at
its deadline. A bounded 90-second reproduction showed
`ray_serve_deployment_queued_queries=1` at the
proxy while `ray_serve_autoscaling_total_requests=0` at the controller; no
worker demand appeared. These runs prove deadline behavior, not recovery from
that forced permanent actor death. The guarded test now explicitly sets
`no_restart=False` and checks for a new process under the same actor ID; a
separate scaled-to-zero test is added but has not yet passed against the
cluster. The fault injection uses the in-pod interpreter because local Ray
Client under `uv run` automatically uploaded its working directory as a
runtime environment.

After restoring the validation RayCluster, a standalone execution scheduled
through public HTTP succeeded once the CPU worker image was pulled. The worker
pull took over four minutes, exceeding the previous 180-second test polling
limit but not the execution's 600-second deadline. With the worker ready, the
public HTTP scheduled/immediate result-comparison test and the RationAI SDK
scheduled/immediate test both passed. The deployed test polling limit is now
630 seconds so it observes the execution's actual terminal outcome on a cold
node.

The local public-API compatibility test sends legacy model and heatmap-builder
POST requests through the real scheduler ingress and HTTP proxy, with only Ray
Serve mocked. It checks the binary body and response headers as well as the
heatmap-builder query and text response. The RationAI SDK's existing model
test checks that its compressed image request uses the same public origin.

The local `test_validation_raw_inference` integration test also passes through
the RationAI SDK, scheduler ingress, and a real Ray Serve validation classifier.
It checks a compressed image request and the classifier's JSON prediction.
The validation Helm profile now declares that classifier at
`/prostate-classifier-1`, but the running release does not include it until a
new validation image is built and deployed. This tests the raw inference call
path, not production model accuracy.

GPU starvation followed by recovery and real legacy model inference remain
target-cluster release checks. The CPU validation RayService does not deploy
the GPU model applications, so these tests cannot prove those paths.
