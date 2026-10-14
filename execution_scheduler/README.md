# Execution scheduler

Durable executions of builder invocations. `ExecutionScheduler` admits,
reads and cancels; its worker claims due work and moves it through processing
and provider finish.

| Module | Owns |
| --- | --- |
| `scheduler.py` | Admission (validates builder, model and provider), views, cancellation, worker lifetime |
| `worker.py` | Claim loop (up to `max_concurrent` claims), lease renewal, dispatch, settlement |
| `transitions.py` | Every state change: admit, claim, renew, settle, cancel, lease recovery, deadlines |
| `domain.py`, `contracts.py` | Internal records and public request/view types |
| `repository.py` | The persistence interface |
| `persistence/postgres.py` | PostgreSQL implementation; a per-model advisory lock serializes capacity checks |
| `persistence/migrations/` | Yoyo migrations, applied at startup ([guide](persistence/migrations/README.md)) |
| `persistence/protected_context.py` | AES-GCM encryption of provider secrets at rest |
| `providers.py` | Provider interface: identify, prepare, finish |

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
  until the lease would have expired; then the attempt stops and another
  claim recovers it. Every write checks the fencing token, so a stale worker
  cannot overwrite a newer attempt.

Tests: `tests/test_execution_scheduler.py` (behaviour, in-memory repository),
`tests/repository_contract.py` (shared by both repositories) and
`tests/test_postgres_execution_repository.py` (PostgreSQL concurrency and
migrations; set `TEST_POSTGRES_DSN`).
