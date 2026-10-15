"""Per-replica limits and wiring, supplied by the builder's Ray `user_config`."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class WorkerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    num_threads: int = Field(default=2, gt=0)
    max_concurrent_tasks: int = Field(default=4, gt=0)
    shared_storage_roots: tuple[Path, ...] = ()


class DeploymentSettings(WorkerSettings):
    builder_id: str = Field(min_length=1)
