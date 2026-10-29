"""Declare the available builders.

Declarations import only Pydantic so the facade can validate requests and list
builders without the slide-processing dependencies. Implementations are loaded
lazily from `implementation` inside Ray replicas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from model_service.builders.contracts import (
    InferenceContract,
    InvalidInvocation,
    OutputKind,
    UnknownBuilderError,
)


if TYPE_CHECKING:
    from collections.abc import Mapping

    from model_service.builders.contracts import JsonValue


@dataclass(frozen=True, slots=True)
class InputSpec:
    required: bool = True
    media_type: str | None = None
    empaia: Mapping[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OutputSpec:
    kind: OutputKind
    media_type: str | None = None
    default_path: str | None = None
    empaia: Mapping[str, JsonValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BuilderSpec:
    id: str
    accepts: InferenceContract
    inputs: Mapping[str, InputSpec]
    outputs: Mapping[str, OutputSpec]
    configuration: type[BaseModel]
    implementation: str

    def resolve_configuration(
        self, overrides: Mapping[str, JsonValue] | None = None
    ) -> BaseModel:
        """Apply caller or provider overrides to the builder's current defaults."""
        try:
            return self.configuration.model_validate(dict(overrides or {}))
        except ValidationError as error:
            # Field names only: values may come from callers.
            fields = sorted({".".join(map(str, e["loc"])) for e in error.errors()})
            raise InvalidInvocation(
                f"{self.id} configuration is invalid: {', '.join(fields)}"
            ) from error


class ScoreEncoding(StrEnum):
    UINT8 = "uint8"
    FLOAT32 = "float32"


_STRIDE = Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)]


class HeatmapConfiguration(BaseModel):
    """`uint8` maps scores in [0, 1] to 0-255 and matches the existing artifacts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stride_fraction: _STRIDE = 0.5
    output_tile_height: Annotated[int, Field(strict=True, gt=0)] = 256
    output_tile_width: Annotated[int, Field(strict=True, gt=0)] = 256
    encoding: ScoreEncoding = ScoreEncoding.UINT8
    # Skip tiles without tissue; they score 0.
    skip_background: bool = True


class EmbeddingConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    stride_fraction: _STRIDE = 1.0


_SLIDE = InputSpec(media_type="application/vnd.openmicroscopy", empaia={"type": "wsi"})
_REGION_OF_INTEREST = InputSpec(
    required=False,
    media_type="image/tiff",
    empaia={
        "type": "rectangle",
        "reference": "slide",
        "classes": ["org.empaia.global.v1.classes.roi"],
    },
)

BUILDERS: dict[str, BuilderSpec] = {
    spec.id: spec
    for spec in (
        BuilderSpec(
            id="heatmap",
            accepts=InferenceContract.SPATIAL_SCORES,
            inputs={"slide": _SLIDE, "region_of_interest": _REGION_OF_INTEREST},
            outputs={
                "heatmap": OutputSpec(
                    kind=OutputKind.SHARED_ARTIFACT,
                    media_type="image/tiff",
                    default_path="heatmap.tiff",
                    empaia={"type": "continuous_pixelmap", "reference": "slide"},
                )
            },
            configuration=HeatmapConfiguration,
            implementation="model_service.builders.heatmap.builder:HeatmapBuilder",
        ),
        BuilderSpec(
            id="tile-embeddings",
            accepts=InferenceContract.TILE_EMBEDDING,
            inputs={"slide": _SLIDE, "region_of_interest": _REGION_OF_INTEREST},
            outputs={
                "embeddings": OutputSpec(
                    kind=OutputKind.SHARED_ARTIFACT,
                    media_type="application/x-hdf5",
                    default_path="embeddings.h5",
                )
            },
            configuration=EmbeddingConfiguration,
            implementation="model_service.builders.embeddings.builder:EmbeddingBuilder",
        ),
    )
}


def get_builder(builder_id: str) -> BuilderSpec:
    try:
        return BUILDERS[builder_id]
    except KeyError as error:
        raise UnknownBuilderError(builder_id) from error
