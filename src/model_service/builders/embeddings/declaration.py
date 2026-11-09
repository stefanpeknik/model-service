"""Tile-embeddings builder declaration: an HDF5 feature table from a `tile-embedding` model."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict

from model_service.builders.contracts import (
    Artifact,
    BuilderInputs,
    BuilderOutputs,
    InferenceContract,
    SharedArtifact,
)
from model_service.builders.spec import (
    BuilderSpec,
    RegionOfInterest,
    Slide,
    StrideFraction,
)


class EmbeddingInputs(BuilderInputs):
    slide: Slide
    # Only tiles the mask covers get a row.
    region_of_interest: RegionOfInterest = None


class EmbeddingConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    stride_fraction: StrideFraction = 1.0


class EmbeddingOutputs(BuilderOutputs):
    embeddings: Annotated[
        SharedArtifact,
        Artifact(media_type="application/x-hdf5", default_path="embeddings.h5"),
    ]


SPEC = BuilderSpec(
    id="tile-embeddings",
    accepts=InferenceContract.TILE_EMBEDDING,
    input_model=EmbeddingInputs,
    output_model=EmbeddingOutputs,
    configuration=EmbeddingConfiguration,
    implementation="model_service.builders.embeddings.builder:EmbeddingBuilder",
)
