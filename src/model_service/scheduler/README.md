# Execution scheduler

Durable executions of builder invocations. Raw model requests are not
executions: they go straight to the model's Ray Serve ingress.
`ExecutionScheduler` admits, reads and cancels; its worker claims due work and
moves it through processing and provider finish.

| Module | Owns |
| --- | --- |
| `execution_scheduler.py` | Admission (validates builder, model and provider), views, cancellation, worker lifetime |
| `worker.py`, `running_work.py` | Claim loop, lease renewal, dispatch, settlement |
| `builder_dispatch.py` | Calling a builder replica over HTTP |
| `transitions/` | Every state change: admit, claim, renew, settle, cancel, lease recovery, deadlines |
| `public.py`, `internal_state.py`, `status.py` | Public request/view types, internal records, statuses |
| `protocols.py` | The `Scheduler` interface the API uses |
| `store/repository.py` | The persistence interface |
| `store/postgres/` | PostgreSQL implementation; an advisory lock per model serializes capacity checks |
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

- Each attempt resolves the model's current catalog entry; a model removed or
  made incompatible since admission fails the execution without retry.
- A builder that rejects with 429 is retried without spending an attempt.
- A publication that fails or whose outcome is unknown is not retried; the
  execution fails and the provider gets a failure report.
- Leases are renewed every third of their duration. Renewal errors are retried
  until the lease would have expired; then the attempt stops and the next
  maintenance pass (every poll interval) recovers it. Every write checks the
  fencing token, so a stale worker cannot overwrite a newer attempt.

Tests: `tests/test_execution_scheduler.py` (behaviour, in-memory repository),
`tests/repository_contract.py` (shared by both repositories) and
`tests/test_postgres_execution_repository.py` (PostgreSQL concurrency and
migrations; set `TEST_POSTGRES_DSN`).
