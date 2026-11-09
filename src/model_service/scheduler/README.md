# Execution scheduler

The state rules of durable builder executions. They compute each change of
an execution record and do no storage or network I/O.

| Module | Owns |
| --- | --- |
| `transitions.py` | Every state change: admit, claim, renew, settle, cancel, lease recovery, deadlines |
| `domain.py`, `contracts.py` | Execution records, leases, retry policies, attempt outcomes and safe errors |

## Lifecycle

```text
queued ──claim──► running (processing) ──result──► running (provider finish) ──► succeeded
   ▲                  │ retryable failure / lost lease                │ failure
   └──── backoff ─────┘                                               ▼
                      │ permanent failure, attempts exhausted    failed ──► failure report
cancel ─────────────────────────────────────────────────────► cancelled ──► cancellation report
```

- A builder that rejects for lack of capacity is retried without spending an
  attempt.
- A publication that fails or whose outcome is unknown is not retried; the
  execution fails and the provider gets a failure report.
- A lease that expires is recovered by retrying or failing its attempt. Every
  settlement checks the fencing token, so a stale attempt cannot overwrite a
  newer one.

Tests: `tests/scheduler/test_transitions.py` and
`tests/scheduler/test_retry_policy.py`.
