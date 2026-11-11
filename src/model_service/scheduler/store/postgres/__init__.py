"""Persist executions in PostgreSQL through operation-scoped SQLAlchemy transactions.

Row locks protect record changes; per-model advisory locks serialize capacity checks.
Each operation runs in its own transaction on a pooled connection, and database time
governs leases.
"""

from model_service.scheduler.store.postgres.admission_sql import AdmissionOperations
from model_service.scheduler.store.postgres.attempt_outcome_sql import (
    SettlementOperations,
)
from model_service.scheduler.store.postgres.claim_sql import ClaimOperations
from model_service.scheduler.store.repository import ExecutionRepository


class PostgresExecutionRepository(
    AdmissionOperations,
    ClaimOperations,
    SettlementOperations,
    ExecutionRepository,
):
    """Provide durable replay-safe admission, leased claims, and recovery using the shared state machine.

    Its operations are grouped by module: admission, claims and settlement;
    base holds the connection, maintenance and shared helpers.
    """


__all__ = ["PostgresExecutionRepository"]
