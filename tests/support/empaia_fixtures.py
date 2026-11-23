"""What the App Service answers, shaped as the SDK's typed models require, and a strict fake gateway."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from model_service.formats import score_grid
from model_service.providers.empaia.app_service.gateway import EmpaiaGateway


if TYPE_CHECKING:
    from pathlib import Path

    from rationai.empaia.typing import SlideInfo
    from rationai.empaia.utils.scalar_output import PixelmapTileSet

    from model_service.contracts.json_values import JsonValue
    from model_service.providers.empaia.app_service.gateway import (
        EmpaiaJobMode,
        MutationGuard,
    )
    from model_service.providers.empaia.app_service.types import DirectAccess


PIXELMAP_ID = "11111111-1111-4111-8111-111111111111"
SLIDE_ID = "22222222-2222-4222-8222-222222222222"


def score_body() -> Any:
    """A float primitive output named `score`, as the SDK posts it.

    Returns:
        The SDK's request model, created by job `job-1`.
    """
    from rationai.empaia.typing import primitives

    return primitives.PostFloatPrimitive.model_validate(
        {
            "name": "score",
            "type": "float",
            "value": 0.75,
            "creator_id": "job-1",
            "creator_type": "job",
        }
    )


def pixelmap_body(**overrides: object) -> dict[str, object]:
    """A valid description of a continuous pixelmap, as the gateway sends it."""
    return {
        "name": "heatmap",
        "type": "continuous_pixelmap",
        "element_type": "float32",
        "channel_count": 1,
        "tilesize": 512,
        "min_value": 0.0,
        "max_value": 1.0,
        "reference_id": SLIDE_ID,
        "reference_type": "wsi",
        "creator_type": "job",
        "creator_id": "job-1",
        "levels": [
            {
                "slide_level": 0,
                "position_min_x": 0,
                "position_max_x": 0,
                "position_min_y": 0,
                "position_max_y": 0,
            }
        ],
        **overrides,
    }


def created(body: dict[str, object], id: str = PIXELMAP_ID) -> dict[str, object]:
    """The App Service's answer to creating an output."""
    return {**body, "id": id, "created_at": 1, "updated_at": 1}


def png(pixels: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def slide_info(width: int = 384, height: int = 256, levels: int = 6) -> SlideInfo:
    """A slide of `levels` levels, each half the size of the one before."""
    from rationai.empaia.typing import SlideInfo

    return SlideInfo.model_validate(
        {
            "id": SLIDE_ID,
            "extent": {"x": width, "y": height, "z": 1},
            "num_levels": levels,
            "pixel_size_nm": {"x": 250.0, "y": 250.0},
            "tile_extent": {"x": 256, "y": 256, "z": 1},
            "levels": [
                {
                    "extent": {"x": width // 2**i, "y": height // 2**i, "z": 1},
                    "downsample_factor": float(2**i),
                }
                for i in range(levels)
            ],
            "format": "test",
            "tissue": None,
            "stain": None,
        }
    )


def slide_data(width: int = 384, height: int = 256) -> dict[str, object]:
    """The slide input's data, as the App Service answers `GET /inputs/slide`."""
    return slide_info(width, height).model_dump(mode="json")


def write_score_grid(path: Path, width: int = 384, height: int = 256) -> Path:
    """A one-channel score grid over a slide's level 0, every cell scored 0.5."""
    rows, columns = -(-height // 64), -(-width // 64)
    score_grid.write(
        path,
        score_grid.ScoreGrid(
            values=np.full((1, rows, columns), 0.5, dtype=np.float32),
            scored=np.ones((rows, columns), dtype=bool),
            level=0,
            level_size=(width, height),
            cell_size=64.0,
            mpp=(0.25, 0.25),
        ),
    )
    return path


class StrictGateway(EmpaiaGateway):
    """A gateway whose every call fails the test; fakes override what they serve."""

    @property
    def job_id(self) -> str:
        return "job-1"

    def set_before_mutation(self, callback: MutationGuard) -> None:
        raise AssertionError("unexpected set_before_mutation")

    async def get_configuration(self) -> dict[str, JsonValue]:
        raise AssertionError("unexpected get_configuration")

    async def get_mode(self) -> EmpaiaJobMode:
        raise AssertionError("unexpected get_mode")

    async def has_input(self, name: str) -> bool:
        raise AssertionError(f"unexpected has_input({name!r})")

    async def get_input_data(self, name: str) -> JsonValue:
        raise AssertionError(f"unexpected get_input_data({name!r})")

    async def direct_access(self, name: str) -> DirectAccess | None:
        raise AssertionError(f"unexpected direct_access({name!r})")

    async def slide_info(self, name: str) -> SlideInfo:
        raise AssertionError(f"unexpected slide_info({name!r})")

    async def post_output(self, name: str, body: Any) -> None:
        raise AssertionError(f"unexpected post_output({name!r})")

    async def publish_pixelmap(
        self, name: str, slide_id: str, tile_sets: tuple[PixelmapTileSet, ...]
    ) -> None:
        raise AssertionError(f"unexpected publish_pixelmap({name!r})")

    async def finalize(self) -> None:
        raise AssertionError("unexpected finalize")

    async def report_progress(self, fraction: float) -> None:
        raise AssertionError("unexpected report_progress")

    async def report_failure(self, message: str) -> None:
        raise AssertionError(f"unexpected report_failure({message!r})")

    async def close(self) -> None:
        raise AssertionError("unexpected close")
