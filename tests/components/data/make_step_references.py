"""Write `step_references.npz`: torchvision's results for the shared steps.

The steps claim to match torchvision; these arrays let the tests check that
without torchvision installed. Rerun only when a case changes:

    PYTHONPATH=src uv run python tests/components/data/make_step_references.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms


OUT = Path(__file__).with_name("step_references.npz")
IMAGENET = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
MODES = {
    "bilinear": transforms.InterpolationMode.BILINEAR,
    "bicubic": transforms.InterpolationMode.BICUBIC,
    "nearest": transforms.InterpolationMode.NEAREST,
}


def main() -> None:
    """Compute every reference from one seeded 7 x 10 RGB tile."""
    tile = np.random.default_rng(7).integers(0, 256, (3, 7, 10), dtype=np.uint8)
    image = Image.fromarray(tile.transpose(1, 2, 0))
    arrays = {"tile": tile}
    normalized = transforms.Normalize(*IMAGENET)(transforms.ToTensor()(image))
    arrays["normalize_imagenet"] = normalized.numpy()
    for mode, interpolation in MODES.items():
        for size in (5, 12):
            resized = transforms.Resize(size, interpolation=interpolation)(image)
            arrays[f"resize_{mode}_{size}"] = np.asarray(resized).transpose(2, 0, 1)
    for size in (4, 9, 12):
        cropped = transforms.CenterCrop(size)(torch.from_numpy(tile))
        arrays[f"center_crop_{size}"] = cropped.numpy()
    np.savez_compressed(OUT, **arrays)


if __name__ == "__main__":
    main()
