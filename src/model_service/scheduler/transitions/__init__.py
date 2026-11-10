"""Compute execution state changes without database or network I/O.

The repository applies these immutable decisions for admission, leases,
deadlines, retries, and provider finish. Repositories supply time and make each decision
atomic.
"""

from model_service.scheduler.transitions.admission import AdmissionTransitions
from model_service.scheduler.transitions.attempt_outcomes import (
    AttemptOutcomeTransitions,
)
from model_service.scheduler.transitions.base import (
    ClaimDecision,
    RenewalDecision,
    SettlementAccepted,
    SettlementDecision,
    SettlementRejected,
    SettlementRejection,
)
from model_service.scheduler.transitions.leases import LeaseTransitions


class ExecutionTransitions(
    AdmissionTransitions, LeaseTransitions, AttemptOutcomeTransitions
):
    """Apply the shared state machine with separate processing and provider retry policies.

    Its rules are grouped by module: admission, leases, attempt_outcomes and
    expiry.
    """


__all__ = [
    "ClaimDecision",
    "ExecutionTransitions",
    "RenewalDecision",
    "SettlementAccepted",
    "SettlementDecision",
    "SettlementRejected",
    "SettlementRejection",
]
