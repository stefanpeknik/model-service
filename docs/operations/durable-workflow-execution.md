# Durable workflow execution

The execution scheduler accepts workflow requests, stores them in PostgreSQL,
and dispatches due work to Ray Serve. It is a long-lived deployment. It does
not create a Kubernetes pod for each execution.

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
`module:attribute` callable. The callable receives the EMPAIA API URL, job ID,
and original job credential and returns the gateway used by preparation,
workflow input access, and provider finish. The gateway module must be present
in both the scheduler and Ray worker images.

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

The target-cluster controller-loss, cluster-loss, and GPU-starvation scenarios
remain release validation steps. They cannot validate this local-only change
set until its images and manifests are available to that cluster; do not treat
the local Ray result as evidence for those Kubernetes failure modes.
