"""Convert EMPAIA inputs at the EMPAIA image boundary.

Rectangle masks retain slide geometry and physical pixel resolution (libvips,
loaded lazily).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_service.providers.empaia.gateway import EmpaiaGatewayError
from model_service.providers.empaia.slides import RectangleMaskWriter


if TYPE_CHECKING:
    from pathlib import Path

    from model_service.providers.empaia.slides import (
        Rectangle,
        SlideMetadata,
    )


class VipsRectangleMaskWriter(RectangleMaskWriter):
    """Rasterize the intersection of a rectangle and slide into a pyramidal TIFF region mask."""

    def write(
        self,
        destination: Path,
        slide: SlideMetadata,
        rectangle: Rectangle,
    ) -> None:
        """Clip the rectangle to the slide and save an 8-bit tiled pyramidal mask.

        The mask's resolution derives from the slide's pixel size.

        Args:
            destination: The TIFF file to write.
            slide: The slide the rectangle lies on.
            rectangle: The region, in slide level-0 pixels.

        Raises:
            EmpaiaGatewayError: Not retryable if the rectangle does not overlap the
                slide; retryable for any other failure, including missing libvips.
        """
        try:
            import pyvips

            x, width = _clip(rectangle.x, rectangle.width, slide.width)
            y, height = _clip(rectangle.y, rectangle.height, slide.height)
            if width <= 0 or height <= 0:
                raise EmpaiaGatewayError.invalid(
                    "EMPAIA rectangle input does not overlap the slide"
                )
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
            raise EmpaiaGatewayError.unavailable(
                "EMPAIA rectangle input could not be materialized"
            ) from error


def _clip(start: int, length: int, extent: int) -> tuple[int, int]:
    """Return the part of a span that lies within 0..extent, as its start and length.

    Args:
        start: The span's start.
        length: The span's length.
        extent: The upper bound.

    Returns:
        The clipped start and length; the length is not positive if nothing is left.
    """
    clipped = max(start, 0)
    return clipped, min(start + length, extent) - clipped
