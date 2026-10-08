"""Expose caller-facing durable scheduling contracts.

Start with scheduler.ExecutionScheduler for admission and worker lifetime;
transitions.ExecutionTransitions owns the shared state rules. Persistence
implementations remain separate.
"""

from execution_scheduler.contracts import (
    ExecutionView,
    ScheduleRequest,
    ScheduleResult,
)


__all__ = [
    "ExecutionView",
    "ScheduleRequest",
    "ScheduleResult",
]
