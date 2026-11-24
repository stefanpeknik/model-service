"""The pipeline runner's replica settings, from its Ray `user_config`."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class ReplicaSettings(BaseModel):
    """Per-replica limits and the shared-storage roots requests may name.

    Attributes:
        num_threads: Threads reading tiles, per run.
        max_concurrent_tasks: Tiles in flight, per run.
        shared_storage_roots: Roots a shared-storage request may name.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    num_threads: int = Field(default=2, gt=0)
    max_concurrent_tasks: int = Field(default=4, gt=0)
    shared_storage_roots: tuple[Path, ...] = ()
