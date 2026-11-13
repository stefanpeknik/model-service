"""What a model of each inference contract declares about itself.

Builders and model clients rely on it.

A new kind of model (a new contract) adds a member of `InferenceContract` and its
`_ModelBinding` subclass here (added to `ModelBinding`); once a
builder accepts it, also its client in deployed_models/model_clients/ (added to
`CLIENTS`). A new model of an existing contract needs none of this.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class InferenceContract(StrEnum):
    """The model-ready input and prediction structure a model deployment declares."""

    SPATIAL_SCORES = "spatial-scores"
    TILE_EMBEDDING = "tile-embedding"


class _ModelBinding(BaseModel):
    """A model's declaration: what builders need to drive it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: Annotated[str, Field(min_length=1)]
    tile_size: Annotated[int, Field(strict=True, gt=0)]
    mpp: Annotated[float, Field(gt=0, allow_inf_nan=False)]


class SpatialScoresModel(_ModelBinding):
    """Ray-handle model returning `n_channels x output_tile_size²` scores in [0, 1]."""

    contract: Literal[InferenceContract.SPATIAL_SCORES] = (
        InferenceContract.SPATIAL_SCORES
    )
    output_tile_size: Annotated[int, Field(strict=True, gt=0)]
    n_channels: Annotated[int, Field(strict=True, gt=0)]


class TileEmbeddingModel(_ModelBinding):
    """Ray-handle model returning float32 embeddings for an RGB tile."""

    contract: Literal[InferenceContract.TILE_EMBEDDING] = (
        InferenceContract.TILE_EMBEDDING
    )


type ModelBinding = Annotated[
    SpatialScoresModel | TileEmbeddingModel, Field(discriminator="contract")
]
