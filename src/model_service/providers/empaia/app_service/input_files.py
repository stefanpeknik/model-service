"""Write a job's inputs to local files: slide links and region-of-interest masks."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from model_service.providers.empaia.app_service.gateway import EmpaiaGatewayError
from model_service.providers.empaia.app_service.types import (
    DirectAccess,
    Rectangle,
    SlideMetadata,
)


type RectangleMaskWriter = Callable[[Path, SlideMetadata, Rectangle], None]


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
        directory.mkdir(parents=True, exist_ok=True)
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


def write_rectangle_input(
    writer: RectangleMaskWriter,
    destination: Path,
    slide: SlideMetadata,
    rectangle: Rectangle,
) -> None:
    """Create the destination directory and invoke the mask writer.

    Args:
        writer: Rasterizes the rectangle, usually `write_rectangle_mask`.
        destination: The mask file to write.
        slide: The slide the rectangle lies on.
        rectangle: The region, in slide level-0 pixels.

    Raises:
        EmpaiaGatewayError: As the writer classified it; retryable for any other
            failure, including missing libvips.
    """
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        writer(destination, slide, rectangle)
    except EmpaiaGatewayError:
        raise
    except Exception as error:
        raise EmpaiaGatewayError(
            "EMPAIA rectangle input could not be materialized", retryable=True
        ) from error


def write_rectangle_mask(
    destination: Path, slide: SlideMetadata, rectangle: Rectangle
) -> None:
    """Rasterize the rectangle, clipped to the slide, into a pyramidal TIFF mask.

    The 8-bit tiled mask's resolution derives from the slide's pixel size;
    libvips is loaded lazily.

    Args:
        destination: The TIFF file to write.
        slide: The slide the rectangle lies on.
        rectangle: The region, in slide level-0 pixels.

    Raises:
        EmpaiaGatewayError: Not retryable if the rectangle does not overlap the
            slide.
    """
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
