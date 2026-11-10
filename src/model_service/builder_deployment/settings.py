"""The builder deployment's Ray `user_config`: which builder, and its replica settings."""

from pydantic import Field

from model_service.builders.base import ReplicaSettings


class DeploymentSettings(ReplicaSettings):
    builder_id: str = Field(min_length=1)
