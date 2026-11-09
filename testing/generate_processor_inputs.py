"""Generate synthetic pyramidal slide and region-mask TIFFs for native processor validation.

The fixtures have known 256-pixel geometry and 0.5 micrometer pixel resolution, verified
by reopening them through OpenSlide.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyvips
from ratiopath.openslide import OpenSlide


def write_inputs(directory: Path) -> None:
    """Write a constant-color slide and full-region mask, then verify their geometry, selected level, and physical resolution.

    Args:
        directory: Directory containing definitions or generated validation fixtures.

    Returns:
        None: No return value.

    Raises:
        RuntimeError: The required lifecycle state or operation result is inconsistent.
    """
    directory.mkdir(parents=True, exist_ok=True)
    slide = pyvips.Image.black(256, 256, bands=3).new_from_image([180, 120, 90])
    mask = pyvips.Image.black(256, 256).new_from_image(255)
    for image, name in ((slide, "slide.tiff"), (mask, "roi.tiff")):
        path = directory / name
        image.tiffsave(
            str(path),
            bigtiff=True,
            compression="lzw",
            tile=True,
            tile_width=256,
            tile_height=256,
            xres=2000,
            yres=2000,
            pyramid=True,
        )
        with OpenSlide(str(path)) as opened:
            if opened.level_dimensions[0] != (256, 256):
                raise RuntimeError(f"unexpected extent in {path}")
            if opened.closest_level(0.5) != 0:
                raise RuntimeError(f"unexpected level in {path}")
            if any(abs(value - 0.5) > 0.01 for value in opened.slide_resolution(0)):
                raise RuntimeError(f"unexpected pixel size in {path}")


if __name__ == "__main__":
    write_inputs(Path(sys.argv[1]))
