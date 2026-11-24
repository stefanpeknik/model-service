"""Persist executions in PostgreSQL through operation-scoped SQLAlchemy transactions.

Row locks protect record changes; per-model advisory locks serialize capacity checks.
Each operation runs in its own transaction on a pooled connection, and database time
governs leases.

PostgresExecutionRepository (repository.py) delegates to one operations object
per concern (admission, claims, settlement, maintenance), each built over the
shared PostgresDatabase (base.py).
"""
