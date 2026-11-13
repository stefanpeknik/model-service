"""What a model of each inference contract declares about itself.

The catalog builds one of these per model file; builders and model clients rely on it.

A new kind of model (a new contract) adds a member of `InferenceContract` and its
`_ModelBinding` subclass here (added to `MODEL_CLASSES` and `ModelBinding`); once a
builder accepts it, also its client in deployed_models/model_clients/ (added to
`CLIENTS`). A new model of an existing contract needs none of this.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, ClassVar, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
)


class InferenceContract(StrEnum):
    """The model-ready input and prediction structure a model deployment declares."""

    SPATIAL_SCORES = "spatial-scores"
    TILE_EMBEDDING = "tile-embedding"


class _ModelBinding(BaseModel):
    """A model's catalog declaration: what builders need to drive it.

    A model file gives each field in its `inference` block if the field is in
    `inference_fields`, otherwise in its first deployment's `user_config`, which
    the model implementation also reads.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    inference_fields: ClassVar[tuple[str, ...]] = ()

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

    # The implementation hard-codes these; the file states them for the catalog.
    inference_fields = ("mpp",)

    contract: Literal[InferenceContract.TILE_EMBEDDING] = (
        InferenceContract.TILE_EMBEDDING
    )


type ModelBinding = Annotated[
    SpatialScoresModel | TileEmbeddingModel, Field(discriminator="contract")
]


MODEL_BINDING: TypeAdapter[ModelBinding] = TypeAdapter(ModelBinding)


# The declaration class of each contract.
MODEL_CLASSES: dict[
    InferenceContract, type[SpatialScoresModel | TileEmbeddingModel]
] = {
    model.model_fields["contract"].default: model
    for model in (SpatialScoresModel, TileEmbeddingModel)
}
