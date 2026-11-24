"""Write a job's inputs to local files: slide links and region-of-interest masks."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

from model_service.directories import make_shared_directory
from model_service.providers.empaia.app_service.gateway import EmpaiaGatewayError
from model_service.providers.empaia.app_service.types import DirectAccess, Rectangle


if TYPE_CHECKING:
    from rationai.empaia.typing import SlideInfo


def link_direct_input(direct: DirectAccess, directory: Path) -> Path:
    """Link the shared file into `directory` under its own name and return the link.

    The name is kept because slide readers pick the format by extension, and a
    MIRAX `.mrxs` file is linked together with its same-named data directory.

    Args:
        direct: The shared file and whether direct access is supported.
        directory: The directory to link into; created if missing.

    Returns:
        The link to the file.

    Raises:
        EmpaiaGatewayError: Not retryable if the input does not support direct
            access; retryable if the file is missing or linking fails.
    """
    if not direct.supported:
        raise EmpaiaGatewayError(
            "EMPAIA input does not support direct access", retryable=False
        )
    source = direct.storage_path
    try:
        if not source.is_file():
            raise EmpaiaGatewayError(
                "EMPAIA input is unavailable on shared storage", retryable=True
            )
        make_shared_directory(directory)
        link = directory / source.name
        link.symlink_to(source)
        companion = source.with_suffix("")
        if source.suffix.lower() == ".mrxs" and companion.is_dir():
            (directory / companion.name).symlink_to(companion, target_is_directory=True)
        return link
    except OSError as error:
        raise EmpaiaGatewayError(
            "EMPAIA input could not be linked from shared storage", retryable=True
        ) from error


def write_rectangle_mask(
    destination: Path, slide: SlideInfo, rectangle: Rectangle
) -> None:
    """Rasterize the rectangle, clipped to the slide, into a pyramidal TIFF mask.

    The 8-bit tiled mask's resolution derives from the slide's pixel size;
    libvips is loaded lazily.

    Args:
        destination: The TIFF file to write.
        slide: The slide the rectangle lies on.
        rectangle: The region, in slide level-0 pixels.

    Raises:
        EmpaiaGatewayError: Not retryable if the slide's size or pixel size is
            not positive and finite, or the rectangle does not overlap the
            slide; retryable for any other failure, including missing libvips.
    """
    sizes = (
        slide.extent.x,
        slide.extent.y,
        slide.pixel_size_nm.x,
        slide.pixel_size_nm.y,
    )
    if not all(math.isfinite(size) and size > 0 for size in sizes):
        # The SDK's model doesn't bound these; a bad slide must not be retried.
        raise EmpaiaGatewayError.invalid(
            "EMPAIA AppService returned an invalid response"
        )
    try:
        make_shared_directory(destination.parent)
        import pyvips

        x, width = _clip(rectangle.x, rectangle.width, slide.extent.x)
        y, height = _clip(rectangle.y, rectangle.height, slide.extent.y)
        if width <= 0 or height <= 0:
            raise EmpaiaGatewayError.invalid(
                "EMPAIA rectangle input does not overlap the slide"
            )
        mask = pyvips.Image.black(slide.extent.x, slide.extent.y)
        selected = pyvips.Image.black(width, height).new_from_image(255)
        mask = mask.insert(selected, x, y)
        mask.tiffsave(
            str(destination),
            bigtiff=True,
            compression="deflate",
            tile=True,
            tile_width=512,
            tile_height=512,
            xres=1_000_000 / slide.pixel_size_nm.x,
            yres=1_000_000 / slide.pixel_size_nm.y,
            pyramid=True,
        )
    except EmpaiaGatewayError:
        raise
    except Exception as error:
        raise EmpaiaGatewayError(
            "EMPAIA rectangle input could not be materialized", retryable=True
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
