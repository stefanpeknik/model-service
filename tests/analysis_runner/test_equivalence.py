"""Heatmap analyses through the analysis runner against today's heatmap processor.

`prostate-heatmap` scores a tile with one number, `episeg-heatmap` with a
float16 map as large as the tile. Both runs use the same fake slide and the
same fake model. With one tile in flight the score grids are identical byte
for byte. At today's concurrency the float32 sums over overlapping tiles
happen in completion order, so a few cells may differ in the last bit, as two
runs of today's code do.
"""

from __future__ import annotations

import asyncio
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest

from model_service.analysis_runner import Concurrency, run_analysis
from model_service.builders.base import BuilderSettings
from model_service.builders.implementations.heatmap.processor import build_heatmap
from model_service.builders.implementations.heatmap.spec import HeatmapConfiguration
from model_service.catalog import load_catalog
from model_service.catalog.served_models import legacy_config
from model_service.contracts.model_declarations import SpatialScoresModel
from model_service.testing import fake_model, fake_slide


if TYPE_CHECKING:
    from collections.abc import Callable

    from model_service.testing import FakeModel


CATALOG = Path(__file__).resolve().parents[2] / "catalog"


class ScoreMapModel:
    """Scores each pixel by its mean brightness, as float16 like EPISeg.

    The map follows the tile's pixels, so a grid with x and y swapped or a
    map placed at the wrong scale differs from today's.
    """

    async def infer(self, tile: np.ndarray) -> np.ndarray:
        return (tile.mean(axis=0, dtype=np.float32) / 255).astype(np.float16)[None]


class JitteryModel:
    """The fake model behind both interfaces, answering after a tile-dependent delay.

    The delay reorders completions when several tiles are in flight.
    """

    def __init__(self, model: FakeModel | ScoreMapModel) -> None:
        self.model = model

    async def infer(self, tile: np.ndarray) -> np.ndarray:
        await asyncio.sleep(zlib.crc32(tile.tobytes()) % 5 / 1000)
        return await self.model.infer(tile)

    predict = infer


def tissue_slide(path: Path) -> Path:
    """3584 x 2560 px at 0.5 µm/px: two noisy tissue blobs on noisy glass."""
    rng = np.random.default_rng(7)
    shape = (2560, 3584)
    y, x = np.mgrid[: shape[0], : shape[1]]
    tissue = ((x - 900) / 300) ** 2 + ((y - 900) / 250) ** 2 < 1
    tissue |= ((x - 2200) / 250) ** 2 + ((y - 1500) / 300) ** 2 < 1
    pixels = np.clip(rng.normal(242, 1, (*shape, 3)), 0, 255)
    stain = np.array([190, 110, 165]) + rng.normal(0, 25, (*shape, 3))
    pixels[tissue] = np.clip(stain[tissue], 0, 255)
    return fake_slide(path, pixels.astype(np.uint8), mpp=0.5)


def roi_mask(path: Path) -> Path:
    """A 512 x 384 mask at 2 µm/px selecting the slide's left half."""
    pixels = np.zeros((384, 512, 3), np.uint8)
    pixels[40:330, 60:250] = 255
    return fake_slide(path, pixels, mpp=2.0)


@dataclass(frozen=True)
class Case:
    """An analysis and the served model today's heatmap builder ran it with.

    Attributes:
        analysis: The analysis ID.
        model: The served-model ID.
        fake: Builds a fake of the served model with its output geometry.
    """

    analysis: str
    model: str
    fake: Callable[[], FakeModel | ScoreMapModel]

    def client(self) -> JitteryModel:
        """A fresh fake of the served model, answering out of order."""
        return JitteryModel(self.fake())


CASES = [
    Case(
        "prostate-heatmap",
        "prostate-classifier-1",
        lambda: fake_model(tile_size=512, mpp=0.468),
    ),
    Case("episeg-heatmap", "episeg-1", ScoreMapModel),
]


async def old_heatmap(
    case: Case,
    slide: Path,
    roi: Path | None,
    out: Path,
    client: JitteryModel,
    settings: BuilderSettings,
) -> Path:
    served = load_catalog(CATALOG).entries("served-models")[case.model].value
    # Today's API binds the model with these fields; the processor reads them.
    binding = SpatialScoresModel(id=case.model, **legacy_config(served))
    client.declaration = binding  # type: ignore[attr-defined]
    await build_heatmap(
        slide,
        roi,
        out,
        client=client,  # type: ignore[arg-type]
        num_threads=settings.num_threads,
        max_concurrent_tasks=settings.max_concurrent_tasks,
        report_progress=lambda _: None,
        **HeatmapConfiguration().model_dump(),
    )
    return out


async def new_heatmap(
    case: Case,
    slide: Path,
    roi: Path | None,
    out: Path,
    model: JitteryModel,
    concurrency: Concurrency,
) -> Path:
    analysis = load_catalog(CATALOG).entries("analyses")[case.analysis].value
    inputs = {"slide": slide} | ({"region_of_interest": roi} if roi else {})
    written = await run_analysis(
        analysis,
        inputs,
        {"heatmap": out},
        {case.model: model},
        concurrency=concurrency,
    )
    return written["heatmap"]


def arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key] for key in stored.files}


@pytest.fixture(scope="module")
def slide(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tissue_slide(tmp_path_factory.mktemp("slide") / "slide.tiff")


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.analysis)
@pytest.mark.parametrize("with_roi", [False, True], ids=["whole slide", "with roi"])
def test_one_tile_in_flight_gives_todays_score_grid_byte_for_byte(
    tmp_path: Path, slide: Path, case: Case, with_roi: bool
) -> None:
    roi = roi_mask(tmp_path / "roi.tiff") if with_roi else None
    model = case.client()

    old = asyncio.run(
        old_heatmap(
            case,
            slide,
            roi,
            tmp_path / "old.npz",
            model,
            BuilderSettings(num_threads=1, max_concurrent_tasks=1),
        )
    )
    new = asyncio.run(
        new_heatmap(
            case,
            slide,
            roi,
            tmp_path / "new.npz",
            model,
            Concurrency(read_threads=1, in_flight=1),
        )
    )

    expected, actual = arrays(old), arrays(new)
    assert expected["scored"].sum() > 10
    assert 0 < expected["scored"].mean() < 1
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        assert (actual[key].dtype, actual[key].shape) == (value.dtype, value.shape), key
        assert actual[key].tobytes() == value.tobytes(), key


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.analysis)
def test_todays_concurrency_matches_within_the_last_bit(
    tmp_path: Path, slide: Path, case: Case
) -> None:
    today = BuilderSettings()
    model = case.client()

    old = asyncio.run(
        old_heatmap(case, slide, None, tmp_path / "old.npz", model, today)
    )
    new = asyncio.run(
        new_heatmap(case, slide, None, tmp_path / "new.npz", model, Concurrency())
    )

    assert Concurrency() == Concurrency(
        read_threads=today.num_threads, in_flight=today.max_concurrent_tasks
    )

    expected, actual = arrays(old), arrays(new)
    np.testing.assert_allclose(actual["values"], expected["values"], rtol=0, atol=1e-7)
    for key in ("scored", "level", "level_size", "cell_size", "mpp"):
        assert actual[key].tobytes() == expected[key].tobytes(), key
