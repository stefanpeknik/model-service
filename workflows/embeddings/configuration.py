"""Validate shared per-execution settings for both embedding workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from workflows.configuration import validate_resolved_configuration


if TYPE_CHECKING:
    from collections.abc import Mapping


class EmbeddingConfiguration(BaseModel):
    """Specify an artifact destination and tile overlap without selecting a model.

    Attributes:
        output_path: Artifact path relative to the invocation's permitted output root.
        stride_fraction: Tile step divided by input tile size, greater than zero to one.
    """

    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)

    output_path: str = Field(default="embeddings.h5", min_length=1)
    stride_fraction: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)


def validate_configuration(value: Mapping[str, object]) -> EmbeddingConfiguration:
    """Require every resolved embedding setting and reject extra or invalid values.

    Args:
        value: Complete configuration after catalog defaults and caller overrides.

    Returns:
        EmbeddingConfiguration: Frozen, validated execution settings.

    Raises:
        InvalidWorkflowInvocation: Configuration is incomplete or invalid.
    """
    return validate_resolved_configuration(
        EmbeddingConfiguration, value, workflow_name="embedding"
    )
