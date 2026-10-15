# Execution scheduler

Durable executions of builder invocations and of single raw model requests
("inference" operations). `ExecutionScheduler` admits, reads and cancels; its
worker claims due work and moves it through processing and provider finish.

| Module | Owns |
| --- | --- |
| `scheduler.py` | Admission (validates builder, model and provider), views, cancellation, worker lifetime |
| `worker.py` | Claim loop with separate builder and tile slots, lease renewal, dispatch, settlement |
| `inference.py` | Raw-request target interface, replay identity and HTTP status classification |
| `transitions.py` | Every state change: admit, claim, renew, settle, cancel, lease recovery, deadlines |
| `domain.py`, `contracts.py` | Internal records and public request/view types |
| `repository.py` | The persistence interface |
| `persistence/postgres.py` | PostgreSQL implementation; an advisory lock per operation and model serializes capacity checks; tile bytes live in `inference_payloads` |
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
- An inference execution succeeds as soon as its model response is stored;
  it has no provider finish and its failure or cancellation is not reported
  anywhere. Finished inference executions are deleted after their retention.
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
