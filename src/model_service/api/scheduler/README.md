# Execution scheduler

Durable executions of analyses (and, until they retire, builder runs). Raw
model requests are not executions: they go straight to the model's Ray Serve
ingress. An execution's target (`targets.py`) is its analysis ID, the served
models it calls, and for an analysis admitted as one the resolved document
and its digest; every attempt sends that document to the analysis runner.
`ExecutionScheduler` admits, reads and cancels; its worker claims due work and
moves it through processing and provider finish.

| Module | Owns |
| --- | --- |
| `execution_scheduler.py` | Admission (validates builder, model and provider), views, cancellation, worker lifetime |
| `worker.py`, `running_work.py` | Claim loop, lease renewal, dispatch, settlement |
| `builder_dispatch.py` | Calling a builder replica over HTTP |
| `transitions/` | Every state change: admit, claim, renew, settle, cancel, lease recovery, deadlines. `ExecutionTransitions` holds one object per concern (`admission`, `leases`, `outcomes`, `expiry`); leases and outcomes get the expiry object for deadline checks |
| `public.py`, `internal_state.py`, `status.py` | Public request/view types, internal records, statuses |
| `protocols.py` | The `Scheduler` interface the API uses |
| `store/repository.py` | The persistence interface |
| `targets.py` | What an execution runs, its capacity keys, and the builder × model to analysis ID mapping (transitional) |
| `store/postgres/` | PostgreSQL implementation; an advisory lock per capacity key (served model, or a model-free analysis), tried in sorted order, serializes capacity checks. `PostgresExecutionRepository` delegates to one operations class per concern (admission, claims, settlement, maintenance), each built over `PostgresDatabase` (pool, transactions, row mapping) |
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

- An analysis attempt sends the document stored at admission, whatever the
  catalog holds now. A builder attempt resolves the model's current catalog
  entry; a model removed or made incompatible since admission fails the
  execution without retry.
- A processing attempt is claimed only when each of its capacity keys has a
  free slot: every served model it calls (`serving.max_concurrent_runs`,
  default `PER_MODEL_LIMIT`), or its analysis ID when it calls none.
- A runner or builder that rejects with 429 is retried without spending an attempt.
- A publication that fails or whose outcome is unknown is not retried; the
  execution fails and the provider gets a failure report.
- Leases are renewed every third of their duration. Renewal errors are retried
  until the lease would have expired; then the attempt stops and the next
  maintenance pass (every poll interval) recovers it. Every write checks the
  fencing token, so a stale worker cannot overwrite a newer attempt.

Tests: `tests/api/scheduler/test_execution_scheduler.py` (behaviour, in-memory repository),
`tests/support/repository_contract.py` (shared by both repositories) and
`tests/api/scheduler/test_postgres_execution_repository.py` (PostgreSQL concurrency and
migrations; set `TEST_POSTGRES_DSN`).
