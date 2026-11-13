"""Heatmap builder declaration: a score grid per slide from a `spatial-scores` model."""

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
)
from model_service.contracts.model_declarations import InferenceContract
from model_service.contracts.result import SharedArtifact


class HeatmapInputs(BuilderInputs):
    slide: Slide
    # Only tiles the mask covers are scored.
    region_of_interest: RegionOfInterest = None


class HeatmapConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    stride_fraction: StrideFraction = 0.5
    # Skip tiles without tissue; they score 0.
    skip_background: bool = True


class HeatmapOutputs(BuilderOutputs):
    heatmap: Annotated[
        SharedArtifact,
        # The score grid format, see `contracts/score_grid.py`.
        Artifact(media_type="application/x-npz", default_path="heatmap.npz"),
    ]


SPEC = BuilderSpec(
    id="heatmap",
    accepts=InferenceContract.SPATIAL_SCORES,
    input_model=HeatmapInputs,
    output_model=HeatmapOutputs,
    configuration=HeatmapConfiguration,
    implementation="model_service.builders.heatmap.builder:HeatmapBuilder",
)
