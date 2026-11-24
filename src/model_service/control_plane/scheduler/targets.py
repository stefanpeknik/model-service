"""What an execution runs: a pipeline, its digest and the models it calls.

An execution stores the pipeline document resolved at admission; every
attempt sends that document to the pipeline runner.

Capacity is counted per model; a pipeline calling none is counted
under its own ID.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable

    from model_service.catalog.pipelines import ResolvedPipeline
    from model_service.contracts.json_values import JsonObject


class CapacityKind(StrEnum):
    """What a capacity key counts: runs using a model, or a model-free pipeline."""

    MODEL = "model"
    PIPELINE = "pipeline"


@dataclass(frozen=True, slots=True, order=True)
class CapacityKey:
    """One slot pool executions are counted against.

    Attributes:
        kind: A model, or a model-free pipeline.
        id: The model's or the pipeline's ID.
    """

    kind: CapacityKind
    id: str

    @property
    def lock_name(self) -> str:
        """The name hashed into the key's advisory lock."""
        return f"capacity:{self.kind}:{self.id}"


@dataclass(frozen=True, slots=True)
class ExecutionTarget:
    """An execution's identity: which pipeline runs, as admitted.

    Attributes:
        pipeline_id: The pipeline ID.
        models: The models it calls, in step order; empty for none.
        pipeline_digest: The resolved document's digest.
        pipeline: The resolved document every attempt sends to the pipeline
            runner.
    """

    pipeline_id: str
    models: tuple[str, ...]
    pipeline_digest: str
    pipeline: JsonObject

    @classmethod
    def of(cls, resolved: ResolvedPipeline) -> ExecutionTarget:
        """The target of a pipeline resolved at admission.

        Args:
            resolved: The pipeline with the run's settings, and its document.

        Returns:
            The target carrying the document and its digest.
        """
        return cls(
            pipeline_id=resolved.pipeline.id,
            models=tuple(resolved.pipeline.models),
            pipeline_digest=resolved.digest,
            pipeline=resolved.document,
        )

    @property
    def capacity_keys(self) -> tuple[CapacityKey, ...]:
        """The pools an attempt takes a slot in, sorted (the lock order)."""
        return capacity_keys(self.pipeline_id, self.models)


def capacity_keys(pipeline_id: str, models: tuple[str, ...]) -> tuple[CapacityKey, ...]:
    """The pools a run of a pipeline takes a slot in, sorted (the lock order).

    Args:
        pipeline_id: The pipeline, which counts as its own pool without models.
        models: The models it calls.

    Returns:
        One key per model, or the pipeline's key when it calls none.
    """
    if not models:
        return (CapacityKey(CapacityKind.PIPELINE, pipeline_id),)
    return tuple(sorted({CapacityKey(CapacityKind.MODEL, model) for model in models}))


type CapacityLimits = Callable[[CapacityKey], int]
"""The slots of a capacity key: how many attempts may hold one at once."""
