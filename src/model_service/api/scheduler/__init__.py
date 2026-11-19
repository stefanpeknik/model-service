"""Durable executions: admission, leases, retries, deadlines and provider finish.

- public.py: what callers and providers see: views, schedule requests,
  admission and outcome types (SubmissionIdentity, ProviderSuccess, ...), errors.
- status.py: execution statuses and error codes.
- transitions/: the state rules every repository applies: admission, leases,
  attempt outcomes (the retry-or-fail decision when an attempt ends) and expiry
  (expired leases, passed deadlines), composed by ExecutionTransitions.
- internal_state.py: internal records, leases, retry policy and outcomes.
"""
