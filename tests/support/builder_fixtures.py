"""Shared fixtures for builder runtime tests: test specs, models and native slides."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from model_service.builders.contracts import (
    BuilderResult,
    InferenceContract,
    InputAccessContext,
    Invocation,
    OutputKind,
    SpatialScoresModel,
    TileEmbeddingModel,
)
from model_service.builders.registry import BuilderSpec, InputSpec, OutputSpec


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping
    from pathlib import Path

    import numpy as np

    from model_service.builders.application import BuilderContext


SPATIAL_MODEL = SpatialScoresModel(
    id="classifier", tile_size=32, output_tile_size=1, n_channels=1, mpp=0.5
)
EMBEDDING_MODEL = TileEmbeddingModel(
    id="encoder", tile_size=224, mpp=0.5, summaries=("tile",)
)


class EmptyConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class StrideConfiguration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    stride_fraction: float = 0.5


def spec(
    *,
    inputs: Mapping[str, InputSpec] | None = None,
    outputs: Mapping[str, OutputSpec] | None = None,
    configuration: type[BaseModel] = EmptyConfiguration,
    accepts: InferenceContract = InferenceContract.SPATIAL_SCORES,
) -> BuilderSpec:
    return BuilderSpec(
        id="test-builder",
        accepts=accepts,
        inputs=inputs or {},
        outputs=outputs or {},
        configuration=configuration,
        implementation="tests.support.builder_fixtures:FunctionBuilder",
    )


def inline_output() -> OutputSpec:
    return OutputSpec(kind=OutputKind.INLINE)


def artifact_output(
    default_path: str = "result.bin", media_type: str | None = None
) -> OutputSpec:
    return OutputSpec(
        kind=OutputKind.SHARED_ARTIFACT,
        media_type=media_type,
        default_path=default_path,
    )


class FunctionBuilder:
    def __init__(
        self, function: Callable[[BuilderContext], Awaitable[BuilderResult]]
    ) -> None:
        self._function = function

    async def run(self, context: BuilderContext) -> BuilderResult:
        return await self._function(context)


def invocation(
    kind: str = "shared_storage",
    state: dict[str, object] | None = None,
    **overrides: object,
) -> Invocation:
    values: dict[str, object] = {
        "inputs": {},
        "resolved_configuration": None,
        "input_access": InputAccessContext(kind, state or {}),
        "deadline_at": datetime.now(UTC) + timedelta(minutes=1),
    }
    values.update(overrides)
    return Invocation(**values)  # type: ignore[arg-type]


def write_slide(
    path: Path,
    pixels: np.ndarray,
    *,
    mpp: float = 0.5,
    pyramid: bool = False,
    tile: int = 128,
) -> None:
    """Write an Aperio-style pyramidal TIFF with the given level-0 resolution."""
    import pyvips

    height, width = pixels.shape[:2]
    image = pyvips.Image.new_from_array(pixels)
    image.set_type(
        pyvips.GValue.gstr_type,
        "image-description",
        f"Aperio Image Library v12.0.0\n{width}x{height} |AppMag = 20|MPP = {mpp}",
    )
    image.tiffsave(
        str(path),
        tile=True,
        tile_width=tile,
        tile_height=tile,
        compression="deflate",
        xres=1000 / mpp,
        yres=1000 / mpp,
        pyramid=pyramid,
    )
