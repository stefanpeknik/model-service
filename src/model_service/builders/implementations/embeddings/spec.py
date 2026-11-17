"""Tile-embeddings builder declaration: an HDF5 feature table from a `tile-embedding` model."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, ConfigDict

from model_service.builders.declarations import (
    Artifact,
    BuilderInputs,
    BuilderOutputs,
    BuilderSpec,
    RegionOfInterest,
    Slide,
    StrideFraction,
    TissueFraction,
)
from model_service.contracts.model_declarations import InferenceContract
from model_service.contracts.result import SharedArtifact


class EmbeddingInputs(BuilderInputs):
    """What the tile-embeddings builder reads.

    Attributes:
        slide: The whole-slide image.
        region_of_interest: Optional mask; only tiles it covers are embedded.
    """

    slide: Slide
    # Only tiles the mask covers get a row.
    region_of_interest: RegionOfInterest = None


class EmbeddingConfiguration(BaseModel):
    """How the tile-embeddings builder tiles the slide.

    Attributes:
        stride_fraction: Tile stride as a fraction of the tile size.
        skip_background: Whether tiles without tissue are skipped (no row).
        min_tissue_fraction: Share of a tile that must be tissue to be embedded
            when skipping background; 0 embeds any tile with some tissue.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    stride_fraction: StrideFraction = 1.0
    # Skip tiles without tissue; they get no row.
    skip_background: bool = True
    min_tissue_fraction: TissueFraction = 0.0


class EmbeddingOutputs(BuilderOutputs):
    """What the tile-embeddings builder writes.

    Attributes:
        embeddings: The HDF5 feature table artifact (`.h5`).
    """

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
    implementation="model_service.builders.implementations.embeddings.builder:EmbeddingBuilder",
)
