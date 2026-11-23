"""What the shared steps compute, against fixed values.

`normalize`, `resize` and `center_crop` are checked against torchvision's
results for one small tile, stored in `data/step_references.npz` (made by
`data/make_step_references.py`), so the check runs without torchvision.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from components.steps.center_crop import CenterCrop
from components.steps.normalize import Normalize
from components.steps.resize import Resize
from components.steps.sigmoid import Sigmoid
from components.steps.softmax import Softmax
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


def scores(*channels: float) -> np.ndarray:
    return np.array([[[[c]] for c in channels]], dtype=np.float32)


def test_sigmoid_maps_logits_to_probabilities() -> None:
    out = run_component(Sigmoid, {"output": scores(0.0, math.log(3), -math.log(3))})

    np.testing.assert_allclose(out["output"], scores(0.5, 0.75, 0.25), rtol=1e-6)


def test_sigmoid_saturates_without_overflow() -> None:
    with np.errstate(over="raise"):
        out = run_component(Sigmoid, {"output": scores(-1000.0, 1000.0)})

    np.testing.assert_array_equal(out["output"], scores(0.0, 1.0))


def test_softmax_spreads_probability_over_the_channels() -> None:
    logits = scores(0.0, math.log(2), math.log(5))

    out = run_component(Softmax, {"output": logits})

    np.testing.assert_allclose(out["output"], scores(1 / 8, 2 / 8, 5 / 8), rtol=1e-6)


def test_softmax_temperature_divides_the_logits_first() -> None:
    logits = scores(0.0, 2 * math.log(3))

    out = run_component(Softmax, {"output": logits}, options={"temperature": 2.0})

    np.testing.assert_allclose(out["output"], scores(0.25, 0.75), rtol=1e-6)


def test_softmax_is_per_pixel_and_survives_large_logits() -> None:
    logits = np.array([[[[1000.0, 0.0]], [[0.0, 1000.0]]]], dtype=np.float32)

    out = run_component(Softmax, {"output": logits})

    np.testing.assert_allclose(
        out["output"], np.array([[[[1.0, 0.0]], [[0.0, 1.0]]]]), atol=1e-6
    )
