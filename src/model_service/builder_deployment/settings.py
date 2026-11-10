"""Per-replica limits and wiring, supplied by the builder's Ray `user_config`."""

from pathlib import Path

from pydantic import Field

from model_service.builders.base import BuilderSettings


class ReplicaSettings(BuilderSettings):
    """A builder's settings plus what the replica's providers need."""

    shared_storage_roots: tuple[Path, ...] = ()


class DeploymentSettings(ReplicaSettings):
    builder_id: str = Field(min_length=1)
