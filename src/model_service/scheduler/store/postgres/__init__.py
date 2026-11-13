"""Persist executions in PostgreSQL through operation-scoped SQLAlchemy transactions.

Row locks protect record changes; per-model advisory locks serialize capacity checks.
Each operation runs in its own transaction on a pooled connection, and database time
governs leases.

PostgresExecutionRepository delegates to one operations object per concern
(admission, claims, settlement, maintenance), each built over the shared
PostgresDatabase.
"""

from model_service.scheduler.store.postgres.repository import (
    PostgresExecutionRepository,
)


__all__ = ["PostgresExecutionRepository"]
