"""What the App Service answers, shaped as the SDK's typed models require, and a strict fake gateway."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from model_service.providers.empaia.app_service.gateway import EmpaiaGateway


if TYPE_CHECKING:
    from rationai.empaia.typing import SlideInfo
    from rationai.empaia.utils.scalar_output import PixelmapTileSet

    from model_service.contracts.json_values import JsonValue
    from model_service.providers.empaia.app_service.gateway import (
        EmpaiaJobMode,
        MutationGuard,
    )


PIXELMAP_ID = "11111111-1111-4111-8111-111111111111"
SLIDE_ID = "22222222-2222-4222-8222-222222222222"


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


class StrictGateway(EmpaiaGateway):
    """A gateway whose every call fails the test; fakes override what they serve."""

    def set_before_mutation(self, callback: MutationGuard) -> None:
        raise AssertionError("unexpected set_before_mutation")

    async def get_configuration(self) -> dict[str, JsonValue]:
        raise AssertionError("unexpected get_configuration")

    async def get_mode(self) -> EmpaiaJobMode:
        raise AssertionError("unexpected get_mode")

    async def has_input(self, name: str) -> bool:
        raise AssertionError(f"unexpected has_input({name!r})")

    async def get_primitive_input(self, name: str) -> JsonValue:
        raise AssertionError(f"unexpected get_primitive_input({name!r})")

    async def materialize_input(self, name: str, directory: str) -> str:
        raise AssertionError(f"unexpected materialize_input({name!r})")

    async def post_primitive(
        self, name: str, output_type: str, value: JsonValue
    ) -> None:
        raise AssertionError(f"unexpected post_primitive({name!r})")

    async def slide_info(self) -> SlideInfo:
        raise AssertionError("unexpected slide_info")

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
