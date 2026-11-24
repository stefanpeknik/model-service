"""What an execution runs: an analysis, its digest and the served models it calls.

An execution stores the analysis document resolved at admission; every
attempt sends that document to the analysis runner.

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
        analysis_id: The analysis ID.
        models: The served models it calls, in step order; empty for none.
        analysis_digest: The resolved document's digest.
        analysis: The resolved document every attempt sends to the analysis
            runner.
    """

    analysis_id: str
    models: tuple[str, ...]
    analysis_digest: str
    analysis: JsonObject

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
        return capacity_keys(self.analysis_id, self.models)


def capacity_keys(analysis_id: str, models: tuple[str, ...]) -> tuple[CapacityKey, ...]:
    """The pools a run of an analysis takes a slot in, sorted (the lock order).

    Args:
        analysis_id: The analysis, which counts as its own pool without models.
        models: The served models it calls.

    Returns:
        One key per served model, or the analysis's key when it calls none.
    """
    if not models:
        return (CapacityKey(CapacityKind.ANALYSIS, analysis_id),)
    return tuple(sorted({CapacityKey(CapacityKind.MODEL, model) for model in models}))


type CapacityLimits = Callable[[CapacityKey], int]
"""The slots of a capacity key: how many attempts may hold one at once."""
