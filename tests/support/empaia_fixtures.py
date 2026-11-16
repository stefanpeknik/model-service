"""What the App Service answers, shaped as the SDK's typed models require."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image


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
