"""What callers and providers see of an execution: its submission identity, errors and priority.

Lease fields and protected provider context live in internal_state.py; statuses and
error codes are in status.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


# The integration a scheduled execution belongs to; each provider names itself.
type ProviderName = str


@dataclass(frozen=True, slots=True)
class ExecutionError:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class SubmissionIdentity:
    """Scope a replay key to its provider and distinguish identical requests from conflicting ones."""

    provider: ProviderName
    key: str
    fingerprint: str


class Priority(IntEnum):
    """Order in which due work is claimed; higher first, within each model's capacity.

    Internal for now: providers set it.
    """

    LOW = 0
    NORMAL = 1
    HIGH = 2
