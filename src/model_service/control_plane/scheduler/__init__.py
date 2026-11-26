"""Durable executions: admission, leases, retries, deadlines and provider finish.

- public.py: what callers and providers see: views, schedule requests,
  admission and outcome types (SubmissionIdentity, ProviderSuccess, ...), errors.
- status.py: execution statuses and error codes.
- execution_scheduler.py: ExecutionScheduler, admission and the worker's lifetime.
- worker.py: claims work, runs attempts, renews leases, settles outcomes.
- running_work.py: the attempts running in this process, for prompt cancellation.
- transitions/: the state rules the repository applies: admission, leases,
  attempt outcomes (the retry-or-fail decision when an attempt ends) and expiry
  (expired leases, passed deadlines), composed by ExecutionTransitions.
- internal_state.py: internal records, leases, retry policy and outcomes.
- protocols.py: what the control plane needs from the scheduler (Scheduler); settings.py: process settings.
- dispatch.py: sending a pipeline run to the runner over HTTP, and how a send can end.
- queue.py: which work is due, which attempt is claimed next, what maintenance fixes.
- store/: repository.py (ExecutionRepository: every operation, over a storage),
  storage.py (what a storage must do), postgres/ (PostgresStorage), rows.py,
  migrate.py and migrations/.
"""
