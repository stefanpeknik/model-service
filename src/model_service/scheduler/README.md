# Execution scheduler

Durable executions of builder invocations: the state rules, the persistence
interface and the provider interface. A repository admits, claims, renews and
settles executions under these rules.

| Module | Owns |
| --- | --- |
| `transitions.py` | Every state change: admit, claim, renew, settle, cancel, lease recovery, deadlines |
| `domain.py`, `contracts.py` | Internal records, schedule requests and safe errors |
| `store/repository.py` | The persistence interface |
| `providers.py` | Provider interface: identify, prepare, finish |

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
  write checks the fencing token, so a stale attempt cannot overwrite a newer
  one.

Tests: `tests/scheduler/test_transitions.py`, `tests/scheduler/test_retry_policy.py`
and `tests/support/repository_contract.py` (the repository contract, run against
the in-memory repository).
