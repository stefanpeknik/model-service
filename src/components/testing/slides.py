"""Fake slides and the frame a slide source gives, for component tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from pathlib import Path


def fake_slide(
    path: Path,
    pixels: np.ndarray | None = None,
    *,
    mpp: float = 0.5,
    pyramid: bool = False,
    tile: int = 128,
) -> Path:
    """Write an Aperio-style tiled TIFF that OpenSlide and pyvips can open.

    Args:
        path: Where to write the slide.
        pixels: Level-0 RGB pixels (height, width, 3) as uint8; by default a
            256 x 256 slide of noisy pink tissue on white glass.
        mpp: Microns per pixel at level 0.
        pyramid: Whether to write lower-resolution levels too.
        tile: The TIFF tile edge in pixels.

    Returns:
        The path written.
    """
    import pyvips

    if pixels is None:
        pixels = _tissue_on_glass()
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
    return path


def fake_frame(**overrides: Any) -> dict[str, Any]:
    """The frame row a slide-tile source gives, for sinks under test.

    Field names follow ratiopath's slide rows. The defaults describe a
    256 x 256 slide at 0.5 µm/px read as 64 px tiles with stride 32.

    Args:
        **overrides: Fields to change.

    Returns:
        The frame row.
    """
    return {
        "extent_x": 256,
        "extent_y": 256,
        "tile_extent_x": 64,
        "tile_extent_y": 64,
        "stride_x": 32,
        "stride_y": 32,
        "level": 0,
        "mpp_x": 0.5,
        "mpp_y": 0.5,
        "downsample": 1.0,
    } | overrides


def _tissue_on_glass() -> np.ndarray:
    """A deterministic slide: a pink disc of noisy tissue on white glass."""
    rng = np.random.default_rng(0)
    y, x = np.mgrid[:256, :256]
    tissue = (x - 128) ** 2 + (y - 128) ** 2 < 80**2
    pixels = np.full((256, 256, 3), 245, np.uint8)
    pink = np.array([200, 120, 170]) + rng.normal(0, 12, (256, 256, 3))
    pixels[tissue] = np.clip(pink[tissue], 0, 255).astype(np.uint8)
    return pixels
