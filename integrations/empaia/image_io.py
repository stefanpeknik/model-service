from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

from integrations.empaia.gateway import EmpaiaGatewayError


if TYPE_CHECKING:
    from integrations.empaia.http_gateway import (
        HttpEmpaiaGateway,
        Rectangle,
        SlideLevel,
        SlideMetadata,
    )
    from workflow_execution.contracts import SharedArtifact


class TiffPixelmapPublisher:
    def __init__(self, tile_size: int = 512) -> None:
        if not 256 <= tile_size <= 2048:
            raise ValueError("EMPAIA pixelmap tile size must be between 256 and 2048")
        self._tile_size = tile_size

    async def publish(
        self,
        gateway: HttpEmpaiaGateway,
        name: str,
        artifact: SharedArtifact,
    ) -> None:
        if artifact.media_type != "image/tiff":
            raise _invalid("EMPAIA shared output must be an image/tiff artifact")
        path = Path(artifact.path)
        if not path.is_file():
            raise _invalid("EMPAIA output artifact is unavailable")
        try:
            import pyvips

            image = pyvips.Image.new_from_file(str(path), access="sequential")
        except Exception as error:
            raise _invalid("EMPAIA output artifact is not a readable TIFF") from error
        if image.format != "uchar":
            raise _invalid("EMPAIA heatmap TIFF must contain unsigned 8-bit values")
        slide = await gateway.slide_metadata()
        level = _matching_level(slide.levels, image.width, image.height)
        tiles_x = math.ceil(image.width / self._tile_size)
        tiles_y = math.ceil(image.height / self._tile_size)
        pixelmap_id = await gateway.create_pixelmap(
            name,
            {
                "name": name,
                "type": "continuous_pixelmap",
                "element_type": "float32",
                "channel_count": image.bands,
                "tilesize": self._tile_size,
                "min_value": 0.0,
                "max_value": 1.0,
                "reference_id": slide.identifier,
                "reference_type": "wsi",
                "creator_type": "job",
                "creator_id": gateway.job_id,
                "levels": [
                    {
                        "slide_level": level.index,
                        "position_min_x": 0,
                        "position_max_x": tiles_x - 1,
                        "position_min_y": 0,
                        "position_max_y": tiles_y - 1,
                    }
                ],
            },
        )
        for tile_y in range(tiles_y):
            for tile_x in range(tiles_x):
                x = tile_x * self._tile_size
                y = tile_y * self._tile_size
                width = min(self._tile_size, image.width - x)
                height = min(self._tile_size, image.height - y)
                tile = image.crop(x, y, width, height)
                if width != self._tile_size or height != self._tile_size:
                    tile = tile.embed(
                        0,
                        0,
                        self._tile_size,
                        self._tile_size,
                        extend="black",
                    )
                tile = tile.cast("float") / 255.0
                await gateway.put_pixelmap_tile(
                    pixelmap_id,
                    level.index,
                    tile_x,
                    tile_y,
                    tile.write_to_memory(),
                )


class VipsRectangleMaskWriter:
    def write(
        self,
        destination: Path,
        slide: SlideMetadata,
        rectangle: Rectangle,
    ) -> None:
        try:
            import pyvips

            x = min(max(rectangle.x, 0), slide.width)
            y = min(max(rectangle.y, 0), slide.height)
            width = min(rectangle.width, slide.width - x)
            height = min(rectangle.height, slide.height - y)
            if width <= 0 or height <= 0:
                raise _invalid("EMPAIA rectangle input does not overlap the slide")
            mask = pyvips.Image.black(slide.width, slide.height)
            selected = pyvips.Image.black(width, height).new_from_image(255)
            mask = mask.insert(selected, x, y)
            mask.tiffsave(
                str(destination),
                bigtiff=True,
                compression="deflate",
                tile=True,
                tile_width=512,
                tile_height=512,
                xres=1_000_000 / slide.pixel_size_nm_x,
                yres=1_000_000 / slide.pixel_size_nm_y,
                pyramid=True,
            )
        except EmpaiaGatewayError:
            raise
        except Exception as error:
            raise _unavailable(
                "EMPAIA rectangle input could not be materialized"
            ) from error


def _matching_level(
    levels: tuple[SlideLevel, ...],
    width: int,
    height: int,
) -> SlideLevel:
    return min(
        levels,
        key=lambda level: abs(level.width - width) + abs(level.height - height),
    )


def _invalid(message: str) -> EmpaiaGatewayError:
    return EmpaiaGatewayError(message, retryable=False)


def _unavailable(message: str) -> EmpaiaGatewayError:
    return EmpaiaGatewayError(message, retryable=True)
