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

Scheduled shared-storage artifacts are written under
`.workflow-attempts/<attempt-hash>/` beneath the invocation's storage root.
Read the artifact path from the completed execution result, rather than
constructing it from `resolved_configuration.output_path`. Different attempts
cannot overwrite each other's outputs. Synchronous workflow invocations and
the legacy slide route keep their existing requested destinations. Artifacts
are retained after execution; retention and garbage collection need an
operational policy before sustained use.

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

`/readyz` checks access to the durable execution repository. Ray downtime does
not remove scheduler pods from Service routing, so admission, polling, and
cancellation remain available while the database works. `/healthz/workflows`
reads Ray Serve's route table separately. It returns 503 with status
`unreachable` for transport failures or invalid route responses, and
`missing_applications` with `missing_workflows` for absent declared routes.
Healthy routes return 200 with status `ready`. Scheduling an undeclared
identifier remains a separate 404 `unknown_workflow` response. Admission does
not require a live replica, so scale-from-zero and temporary capacity shortages
remain durable queueing cases.

## EMPAIA adapter

Set `executionScheduler.empaia.enabled`, provide a shared workspace claim,
and configure `executionScheduler.empaia.gatewayFactory` as a
`module:attribute` callable. The callable receives an `EmpaiaJobContext` with
the EMPAIA API URL, job ID, and original job credential. It returns the gateway
used by preparation, workflow input access, and provider finish. The gateway
module must be present in both the scheduler and Ray worker images.

The current direct-access adapter trusts readable paths supplied by AppService
and links them into its workspace. It does not enforce a storage-root or WSI
plugin allow-list. Keep EMPAIA disabled outside the controlled validation
profile until service authentication and storage trust are resolved. A job
credential authenticates outbound AppService requests, not the caller of the
model-service admission endpoint.

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

The full deployed scheduler smoke suite then passed its three normal-capacity
tests; two deliberate zero-worker tests were skipped because worker capacity
was enabled. Both deployed RationAI SDK smoke tests passed. A database check
after the suites found no queued or running validation executions.

The local public-API compatibility test sends legacy model and heatmap-builder
POST requests through the real scheduler ingress and HTTP proxy, with only Ray
Serve mocked. It checks the binary body and response headers as well as the
heatmap-builder query and text response. The RationAI SDK's existing model
test checks that its compressed image request uses the same public origin.

The local `test_validation_raw_inference` integration test also passes through
the RationAI SDK, scheduler ingress, and a real Ray Serve validation classifier.
It checks a compressed image request and the classifier's JSON prediction.
The validation Helm profile declares that classifier at
`/prostate-classifier-1`, but the running CPU RayService does not include it.
The deployed scheduler instead routes inference to the existing production
RayService while keeping workflow dispatch on the validation RayService.

On 2026-09-25, one RationAI SDK `classify_image` request to the existing
production Ray Serve service in `peknik-ns` started its zero-replica GPU
classifier and returned `0.256591796875` for a zero-filled 512 by 512 RGB
tile. The worker was healthy afterward. This proves the original direct Serve
route with the MLflow-backed model, but not the deployed execution scheduler's
forwarding path: the validation scheduler points to a separate CPU RayService.

The deployed EMPAIA contract suite passed all 12 tests against the validation
scheduler and in-namespace mock AppService. It covered admission, replay,
publication, cancellation, failure reporting, and uncertain provider outcomes.
The first test's 180-second observation window expired during a cold worker
image pull; the execution later succeeded and the AppService recorded one tile
upload and one finalization. EMPAIA terminal and status polling now use the
same 630-second window as the deployed workflow tests.

The guarded in-flight Serve-controller restart test passed with
`no_restart=False`. The guarded finalization-ack loss test also passed using
its dedicated mock AppService image: after scheduler replacement, it reported
an unknown provider-finish outcome without repeating output creation, tile
upload, or finalization. The mock AppService deployment was restored to its
original image afterward.

The first deployed SDK inference call through the scheduler failed while the
production GPU classifier scaled from zero: the proxy used HTTPX's default
five-second read timeout. A later call reached Ray Serve but leaked its 307
trailing-slash redirect to the SDK, which could not resolve the internal
service hostname. Public-HTTP regression tests cover both cases. The scheduler
now allows a separate inference origin, uses a configurable 900-second
inference timeout, and follows the internal redirect before responding.
With those changes deployed, one SDK `classify_image` request through the
public scheduler URL returned `0.256591796875` for the same test tile. Both
deployed SDK workflow tests passed again through that URL, covering scheduled
and synchronous invocation plus cancellation. The validation release uses
`model-service-scheduler:split-inference-redirect-20260925-1`.

GPU starvation followed by recovery remains a target-cluster release check.
The CPU validation RayService does not deploy the GPU model applications, so
its own inference tests cannot prove that path.

## First-phase checkpoint entry points

The public caller origin for this validation release is the execution scheduler
service at `model-service-validation-execution-scheduler.peknik-ns.svc.cluster.local:8080`.
The RationAI SDK must receive that URL as `base_url`; its default points to a
different namespace. The scheduler sends workflow calls to the validation Ray
Serve service and model inference to the production Ray Serve service in
`peknik-ns`.

| Caller | Public HTTP path | Result |
| --- | --- | --- |
| RationAI SDK `workflows.run` | `POST /v1/workflows/prostate-heatmap/run` | Synchronous workflow result |
| RationAI SDK `workflows.schedule`, then `executions.get` or `executions.cancel` | `POST /v1/workflows/prostate-heatmap/schedule`, then `GET /v1/executions/{id}` or `POST /v1/executions/{id}/cancel` | Durable execution status |
| RationAI SDK `models.classify_image` | `POST /prostate-classifier-1` | Production classifier prediction |
| EMPAIA backend through the validation mock AppService | `POST /v1/providers/empaia/workflows/prostate-heatmap/schedule` | Durable execution and mock publication |

To repeat the SDK workflow checks from this repository, open a local port
forward and run the public-interface test suite in another terminal:

```sh
oc -n peknik-ns port-forward svc/model-service-validation-execution-scheduler 18080:8080
VALIDATION_ORIGIN=http://127.0.0.1:18080 UV_CACHE_DIR=.cache/uv uv run --no-sync python -m unittest tests.integration.test_deployed_sdk -v
```

The raw classifier call uses a 512 by 512 RGB `uint8` tile and a 900-second
client timeout to allow a cold GPU worker to start. Starting that worker may
read its GitHub runtime archive and the MLflow artifact. Send one request, then
inspect the result and worker state before sending another. The deployed
EMPAIA suite uses only the in-namespace mock AppService; it does not validate
a real EMPAIA installation.

## Checkpoint review fixes, 2026-09-27

The local source now keeps the worker loop alive after an individual claim
loses its lease. Renewal exceptions or a stalled renewal stop local processing
and provider finish, and their late results are not settled. Settlement also
rejects an expired lease even before another worker recovers it. Worker shutdown
still propagates cancellation and stops active local work.

Deadline expiry and abandoned-lease recovery run independently of the worker's
single active claim. This does not increase dispatch concurrency. Backoff
saturates before duration multiplication can overflow. Durable API readiness
now checks the database, while `/healthz/workflows` reports Ray degradation
separately.

The tracked local unit suite plus new retry-policy tests ran 112 tests against
a disposable localhost PostgreSQL 16 container. All passed except one skipped
TIFF-publication test because libvips was unavailable. The suite includes
production PostgreSQL transactions, worker-loop regressions, public HTTP
admission, polling, and cancellation during a mocked Ray outage, and existing
raw-inference and synchronous-workflow compatibility checks. Changed Python
files passed Ruff. This is not evidence of live Kubernetes Service behavior.

These fixes have not been deployed. The validation image recorded above still
uses the earlier readiness and worker behavior. No cluster resources or other
repositories were modified for this follow-up.

EMPAIA service authentication on both admission paths, known publication-failure
reporting, and negative-origin rectangle clipping remain deferred. Maintenance
now advances queued deadlines while work is blocked, but concurrent delivery of
EMPAIA terminal reports while the local worker is occupied remains a provider
follow-up. Do not treat the EMPAIA admission paths as production-ready before
the separate service-authentication finding is resolved.
