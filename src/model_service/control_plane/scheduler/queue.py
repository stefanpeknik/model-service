"""Which work is due, which attempt is claimed next, and what maintenance must fix.

Pure rules over execution records; the repository applies them to what storage
holds, so every storage behaves the same.
"""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

from model_service.control_plane.scheduler.internal_state import WorkKind
from model_service.control_plane.scheduler.transitions.common import work_deadline_at


if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import datetime

    from model_service.control_plane.scheduler.internal_state import ExecutionRecord
    from model_service.control_plane.scheduler.targets import (
        CapacityKey,
        CapacityLimits,
    )
    from model_service.control_plane.scheduler.transitions.expiry import (
        ExpiryTransitions,
    )


# Processing competes in groups: the models it calls, or a model-free pipeline's ID.
type ProcessingGroup = tuple[tuple[str, ...], str | None]


class WorkQueue:
    """The order work is claimed in and the fixes maintenance applies."""

    def __init__(
        self, capacity_limit: CapacityLimits, expiry: ExpiryTransitions
    ) -> None:
        """Keep the pool sizes and the expiry transitions.

        Args:
            capacity_limit: Slots of each capacity key.
            expiry: The deadline and lease-recovery transitions.
        """
        self._capacity_limit = capacity_limit
        self._expiry = expiry

    @staticmethod
    def is_due(execution: ExecutionRecord, now: datetime) -> bool:
        """Whether an execution has unclaimed work whose time came and whose deadline did not pass.

        Args:
            execution: The record.
            now: Current storage time.

        Returns:
            True if a worker may claim it now, capacity permitting.
        """
        return (
            execution.work_kind is not None
            and execution.fencing_token is None
            and execution.next_attempt_at is not None
            and execution.next_attempt_at <= now
            and work_deadline_at(execution) > now
        )

    def pick_claim(
        self, executions: Iterable[ExecutionRecord], now: datetime
    ) -> ExecutionRecord | None:
        """Choose the attempt to claim next.

        Provider work (publication and reports) comes first, by priority and due
        time. Processing competes in groups of executions that call the same
        models (a model-free pipeline by its ID): groups go by their top priority
        and oldest execution, and a group is skipped while any pool it uses is full.

        Args:
            executions: Every execution with work or a lease.
            now: Current storage time.

        Returns:
            The execution to claim, or None when nothing may be claimed.
        """
        executions = list(executions)
        due = [execution for execution in executions if self.is_due(execution, now)]
        provider_work = [e for e in due if e.work_kind is not WorkKind.PROCESSING]
        if provider_work:
            return min(provider_work, key=_attempt_order)
        groups: dict[ProcessingGroup, list[ExecutionRecord]] = {}
        for execution in due:
            groups.setdefault(_group(execution), []).append(execution)
        active = _active_processing(executions, now)
        for group in sorted(
            groups.values(),
            key=lambda g: (-max(e.priority for e in g), min(e.created_at for e in g)),
        ):
            keys = group[0].target.capacity_keys
            if all(active[key] < self._capacity_limit(key) for key in keys):
                return min(group, key=_attempt_order)
        return None

    def maintenance_change(
        self, execution: ExecutionRecord, now: datetime
    ) -> ExecutionRecord | None:
        """Apply a passed deadline, then recover an abandoned lease.

        Args:
            execution: The record.
            now: Current storage time.

        Returns:
            The changed record, or None when maintenance leaves it alone.
        """
        changed = execution
        if changed.work_kind is not None and work_deadline_at(changed) <= now:
            changed = self._expiry.deadline(changed, now)
        if (
            changed.fencing_token is not None
            and changed.lease_expires_at is not None
            and not _lease_live(changed, now)
        ):
            changed = self._expiry.recover_expired(changed, now)
        return None if changed is execution else changed


def _active_processing(
    executions: Iterable[ExecutionRecord], now: datetime
) -> Counter[CapacityKey]:
    """Count the processing attempts holding a live lease, per capacity key.

    Args:
        executions: Records to count over.
        now: Current storage time.

    Returns:
        Slots in use per key.
    """
    counts: Counter[CapacityKey] = Counter()
    for execution in executions:
        if (
            execution.work_kind is WorkKind.PROCESSING
            and execution.fencing_token is not None
            and _lease_live(execution, now)
        ):
            counts.update(execution.target.capacity_keys)
    return counts


def _lease_live(execution: ExecutionRecord, now: datetime) -> bool:
    """Whether the execution's lease runs past now."""
    return execution.lease_expires_at is not None and execution.lease_expires_at > now


def _attempt_order(execution: ExecutionRecord) -> tuple[object, ...]:
    """Higher priority first, then the earliest due, then the oldest."""
    return (-execution.priority, execution.next_attempt_at, execution.created_at)


def _group(execution: ExecutionRecord) -> ProcessingGroup:
    """The group an execution's processing competes in."""
    target = execution.target
    return (target.models, target.pipeline_id if not target.models else None)
