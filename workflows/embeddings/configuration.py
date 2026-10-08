from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from workflow_execution.contracts import InvalidWorkflowInvocation


if TYPE_CHECKING:
    from collections.abc import Mapping


class EmbeddingConfiguration(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    output_path: str = Field(default="embeddings.h5", min_length=1)
    stride_fraction: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)


def validate_configuration(value: Mapping[str, object]) -> EmbeddingConfiguration:
    try:
        config = EmbeddingConfiguration.model_validate(value)
        if config.model_fields_set != EmbeddingConfiguration.model_fields.keys():
            raise ValueError("incomplete configuration")
        return config
    except (ValueError, ValidationError) as error:
        raise InvalidWorkflowInvocation("invalid embedding configuration") from error
