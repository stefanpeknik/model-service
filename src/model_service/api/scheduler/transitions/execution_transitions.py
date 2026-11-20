"""ExecutionTransitions: the state rules of every concern, composed over two retry policies."""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_service.api.scheduler.transitions.admission import AdmissionTransitions
from model_service.api.scheduler.transitions.attempt_outcomes import (
    AttemptOutcomeTransitions,
)
from model_service.api.scheduler.transitions.expiry import ExpiryTransitions
from model_service.api.scheduler.transitions.leases import LeaseTransitions


if TYPE_CHECKING:
    from model_service.api.scheduler.internal_state import RetryPolicy


class ExecutionTransitions:
    """Apply the shared state machine with separate processing and provider retry policies.

    Attributes:
        admission: Admitting and cancelling executions.
        leases: Claiming work and deciding lease renewal.
        outcomes: Settling an attempt's outcome.
        expiry: Expired leases and passed deadlines.
    """

    def __init__(
        self,
        processing_policy: RetryPolicy,
        finish_policy: RetryPolicy,
    ) -> None:
        """Build each group of rules over the two retry policies.

        Args:
            processing_policy: Attempts and backoff for builder processing.
            finish_policy: Attempts and backoff for provider finish and reports.
        """
        self.admission = AdmissionTransitions()
        self.expiry = ExpiryTransitions(processing_policy, finish_policy)
        self.leases = LeaseTransitions(self.expiry)
        self.outcomes = AttemptOutcomeTransitions(
            processing_policy, finish_policy, self.expiry
        )
