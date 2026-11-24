"""What the shared steps compute, against fixed values.

`normalize`, `resize` and `center_crop` are checked against torchvision's
results for one small tile, stored in `data/step_references.npz` (made by
`data/make_step_references.py`), so the check runs without torchvision.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from components.steps.center_crop import CenterCrop
from components.steps.normalize import Normalize
from components.steps.resize import Resize
from tests.components.helpers import run_component


REFERENCES = np.load(Path(__file__).parent / "data" / "step_references.npz")
TILE = REFERENCES["tile"][None]


def test_normalize_matches_torchvision_to_tensor_and_normalize() -> None:
    out = run_component(Normalize, {"tile": TILE}, options={"preset": "imagenet"})

    assert out["tile"].dtype == np.float32
    np.testing.assert_array_equal(out["tile"][0], REFERENCES["normalize_imagenet"])


@pytest.mark.parametrize("interpolation", ["bilinear", "bicubic", "nearest"])
@pytest.mark.parametrize("size", [5, 12])
def test_resize_matches_torchvision(interpolation: str, size: int) -> None:
    options = {"size": size, "interpolation": interpolation}

    out = run_component(Resize, {"tile": TILE}, options=options)

    expected = REFERENCES[f"resize_{interpolation}_{size}"]
    np.testing.assert_array_equal(out["tile"][0], expected)


@pytest.mark.parametrize("size", [4, 9, 12])
def test_center_crop_matches_torchvision(size: int) -> None:
    out = run_component(CenterCrop, {"tile": TILE}, options={"size": size})

    np.testing.assert_array_equal(out["tile"][0], REFERENCES[f"center_crop_{size}"])
