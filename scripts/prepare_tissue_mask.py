from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from openslide import OpenSlide
from PIL import Image, ImageFilter


def tissue_mask(overview: Image.Image) -> Image.Image:
    pixels = np.asarray(overview.convert("RGB"), dtype=np.int16)
    minimum = pixels.min(axis=2)
    spread = pixels.max(axis=2) - minimum
    mean = pixels.mean(axis=2)
    foreground = ((minimum < 250) & (spread >= 10)) | (mean < 220)
    mask = Image.fromarray(np.where(foreground, 255, 0).astype(np.uint8))
    return mask.filter(ImageFilter.MedianFilter(3)).filter(ImageFilter.MaxFilter(11))


def preview(overview: Image.Image, mask: Image.Image) -> Image.Image:
    base = overview.convert("RGBA")
    tint = Image.new("RGBA", base.size, (0, 180, 0, 95))
    tint.putalpha(mask.point(lambda value: 95 if value else 0))
    return Image.alpha_composite(base, tint).convert("RGB")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide", type=Path, required=True)
    parser.add_argument("--max-overview-edge", type=int, default=1600)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", type=Path, required=True)
    args = parser.parse_args()

    for target in (args.output, args.preview):
        if target.exists():
            raise FileExistsError(f"refusing to overwrite {target}")
        target.parent.mkdir(parents=True, exist_ok=True)

    if args.max_overview_edge <= 0:
        raise ValueError("max overview edge must be positive")
    slide = OpenSlide(str(args.slide))
    try:
        level = next(
            (
                index
                for index, dimensions in enumerate(slide.level_dimensions)
                if max(dimensions) <= args.max_overview_edge
            ),
            slide.level_count - 1,
        )
        dimensions = slide.level_dimensions[level]
        region = slide.read_region((0, 0), level, dimensions)
        overview = Image.new("RGB", dimensions, "white")
        overview.paste(region, mask=region.getchannel("A"))
        slide_width, slide_height = slide.dimensions
        slide_mpp_x = float(slide.properties["openslide.mpp-x"])
        slide_mpp_y = float(slide.properties["openslide.mpp-y"])
    finally:
        slide.close()
    mask = tissue_mask(overview)
    width, height = overview.size
    mpp_x = slide_mpp_x * slide_width / width
    mpp_y = slide_mpp_y * slide_height / height
    density = f"{25400 / mpp_x:.6f}x{25400 / mpp_y:.6f}"

    with TemporaryDirectory(dir=args.output.parent) as temporary_directory:
        temporary_mask = Path(temporary_directory) / "mask.png"
        mask.save(temporary_mask)
        subprocess.run(
            [
                "magick",
                str(temporary_mask),
                "-units",
                "PixelsPerInch",
                "-density",
                density,
                "-define",
                "tiff:tile-geometry=256x256",
                "-compress",
                "Zip",
                str(args.output),
            ],
            check=True,
        )
    preview(overview, mask).save(args.preview)
    covered = np.count_nonzero(np.asarray(mask)) / (width * height)
    print(
        f"mask={args.output} level={level} size={width}x{height} coverage={covered:.2%}"
    )
    print(f"preview={args.preview} resolution={mpp_x:.4f}x{mpp_y:.4f} um/px")


if __name__ == "__main__":
    main()
