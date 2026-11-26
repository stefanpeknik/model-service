# Execution scheduler

Durable executions of pipelines. Raw model requests are not executions: they
go straight to the model's Ray Serve ingress. An execution's target
(`targets.py`) is its pipeline ID, the models it calls, and the
resolved document with its digest; every attempt sends that document to the
pipeline runner.
`ExecutionScheduler` admits, reads and cancels; its worker claims due work and
moves it through processing and provider finish.

| Module | Owns |
| --- | --- |
| `execution_scheduler.py` | Admission (resolves the pipeline, validates the provider), views, cancellation, worker lifetime |
| `worker.py`, `running_work.py` | Claim loop, lease renewal, dispatch, settlement |
| `dispatch.py` | Calling the pipeline runner over HTTP |
| `transitions/` | Every state change: admit, claim, renew, settle, cancel, lease recovery, deadlines. `ExecutionTransitions` holds one object per concern (`admission`, `leases`, `outcomes`, `expiry`); leases and outcomes get the expiry object for deadline checks |
| `public.py`, `internal_state.py`, `status.py` | Public request/view types, internal records, statuses |
| `protocols.py` | The `Scheduler` interface the control plane uses |
| `queue.py` | Which work is due, which attempt is claimed next (provider work first, then processing with a free slot in each pool), what maintenance fixes |
| `store/repository.py` | `ExecutionRepository`: every operation (admit, claim, renew, settle, cancel, maintain) as one transaction over a storage, applying `queue.py` and `transitions/`; encrypts provider contexts and drops them once work ends |
| `store/storage.py` | What a storage must do: read and write records in transactions, and serialize claims |
| `targets.py` | What an execution runs, and its capacity keys |
| `store/postgres/` | `PostgresStorage`: rows over a pooled connection; row locks guard each record's changes and one transaction-scoped advisory lock serializes claims |
| `store/migrations/` | Yoyo migrations, applied at startup ([guide](store/migrations/README.md)) |
| `store/protected_context.py` | AES-GCM encryption of provider secrets at rest |

The provider interface (identify, prepare, finish) is `providers/provider.py`.

## Lifecycle

```text
queued ──claim──► running (processing) ──result──► running (provider finish) ──► succeeded
   ▲                  │ retryable failure / lost lease                │ failure
   └──── backoff ─────┘                                               ▼
                      │ permanent failure, attempts exhausted    failed ──► failure report
cancel ─────────────────────────────────────────────────────► cancelled ──► cancellation report
```

- An attempt sends the document stored at admission, whatever the catalog
  holds now.
- A processing attempt is claimed only when each of its capacity keys has a
  free slot: every model it calls (`serving.max_concurrent_runs`,
  default `PER_MODEL_LIMIT`), or its pipeline ID when it calls none.
- A runner that rejects with 429 is retried without spending an attempt.
- A publication that fails or whose outcome is unknown is not retried; the
  execution fails and the provider gets a failure report.
- Leases are renewed every third of their duration. Renewal errors are retried
  until the lease would have expired; then the attempt stops and the next
  maintenance pass (every poll interval) recovers it. Every write checks the
  fencing token, so a stale worker cannot overwrite a newer attempt.

Tests: `tests/control_plane/scheduler/test_execution_scheduler.py` (scheduler and worker behaviour)
and `tests/control_plane/scheduler/test_execution_repository.py` (claim order, capacity, leases,
deadlines, credentials), both on `ExecutionRepository` over the in-memory storage in
`tests/support/memory_repository.py`. No test runs `PostgresStorage` or the migrations.
