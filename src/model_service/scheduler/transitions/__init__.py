"""Compute execution state changes without database or network I/O.

The repository applies these immutable decisions for admission, leases,
deadlines, retries, and provider finish. Repositories supply time and make each decision
atomic.

ExecutionTransitions (execution_transitions.py) composes one object per
concern: admission, leases, outcomes and expiry. Leases and outcomes receive the
expiry object, because a renewal or settlement past its deadline applies the
deadline rule.
"""
