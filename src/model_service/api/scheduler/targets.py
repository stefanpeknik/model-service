"""What an execution runs: an analysis, its digest and the served models it calls.

An execution stores the analysis document resolved at admission; every
attempt sends that document to the analysis runner. Executions admitted
before analyses replaced the builders have none: migration 0009 named them
after the matching analysis, or `<builder>:<model>`, and they can no longer
run.

Capacity is counted per served model; an analysis calling none is counted
under its own ID.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable

    from model_service.catalog.analyses import ResolvedAnalysis
    from model_service.contracts.json_values import JsonObject


class CapacityKind(StrEnum):
    """What a capacity key counts: runs using a served model, or a model-free analysis."""

    MODEL = "model"
    ANALYSIS = "analysis"


@dataclass(frozen=True, slots=True, order=True)
class CapacityKey:
    """One slot pool executions are counted against.

    Attributes:
        kind: A served model, or a model-free analysis.
        id: The served model's or the analysis's ID.
    """

    kind: CapacityKind
    id: str

    @property
    def lock_name(self) -> str:
        """The name hashed into the key's advisory lock."""
        return f"capacity:{self.kind}:{self.id}"


@dataclass(frozen=True, slots=True)
class ExecutionTarget:
    """An execution's identity: which analysis runs, as admitted.

    Attributes:
        analysis_id: The analysis ID, or a legacy `<builder>:<model>` ID.
        models: The served models it calls, in step order; empty for none.
        analysis_digest: The resolved document's digest; None without one.
        analysis: The resolved document every attempt sends to the analysis
            runner; None for an execution admitted before analyses.
    """

    analysis_id: str
    models: tuple[str, ...] = ()
    analysis_digest: str | None = None
    analysis: JsonObject | None = None

    @classmethod
    def of(cls, resolved: ResolvedAnalysis) -> ExecutionTarget:
        """The target of an analysis resolved at admission.

        Args:
            resolved: The analysis with the run's settings, and its document.

        Returns:
            The target carrying the document and its digest.
        """
        return cls(
            analysis_id=resolved.analysis.id,
            models=tuple(resolved.analysis.models),
            analysis_digest=resolved.digest,
            analysis=resolved.document,
        )

    @property
    def capacity_keys(self) -> tuple[CapacityKey, ...]:
        """The pools an attempt takes a slot in, sorted (the lock order)."""
        if not self.models:
            return (CapacityKey(CapacityKind.ANALYSIS, self.analysis_id),)
        return tuple(
            sorted({CapacityKey(CapacityKind.MODEL, model) for model in self.models})
        )


type CapacityLimits = Callable[[CapacityKey], int]
"""The slots of a capacity key: how many attempts may hold one at once."""


def fixed_limit(limit: int) -> CapacityLimits:
    """The same limit for every capacity key.

    Args:
        limit: Slots per key.

    Returns:
        The limits.

    Raises:
        ValueError: The limit is not positive.
    """
    if limit <= 0:
        raise ValueError("per-model limits must be positive")
    return lambda _key: limit
