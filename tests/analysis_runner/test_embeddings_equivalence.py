"""The embeddings analyses through the analysis runner against today's embeddings processor.

Both run on the same fake slide with the same fake encoder, whose features
have the served model's summaries. The `embedding_table` sink takes rows in
grid order, as today's processor writes them, so at today's concurrency the
HDF5 files are identical byte for byte.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from model_service.analysis_runner import Concurrency, run_analysis
from model_service.builders.base import BuilderSettings
from model_service.builders.implementations.embeddings.processor import (
    build_embeddings,
)
from model_service.builders.implementations.embeddings.spec import (
    EmbeddingConfiguration,
)
from model_service.catalog import load_catalog
from model_service.catalog.served_models import legacy_config
from model_service.contracts.model_declarations import TileEmbeddingModel
from model_service.testing import fake_model, fake_slide
from tests.analysis_runner.test_equivalence import JitteryModel, roi_mask, tissue_slide


CATALOG = Path(__file__).resolve().parents[2] / "catalog"
ANALYSES = {
    "virchow2-embeddings": "virchow2",
    "prov-gigapath-embeddings": "prov-gigapath",
}


class JitteryEncoder(JitteryModel):
    """The jittery fake model, also behind today's `embed`."""

    async def embed(self, tile: np.ndarray) -> np.ndarray:
        return await self.infer(tile)


def encoder(model_id: str) -> JitteryEncoder:
    """A fake encoder with the served model's geometry and summaries."""
    served = load_catalog(CATALOG).entries("served-models")[model_id].value
    model = fake_model(
        {"kind": "embedding", "dim": 16, "summaries": served.output.summaries},
        tile_size=served.input.tile_size,
        mpp=served.input.mpp,
    )
    client = JitteryEncoder(model)
    # Today's API binds the model with these fields; the processor reads them.
    client.declaration = TileEmbeddingModel(id=model_id, **legacy_config(served))  # type: ignore[attr-defined]
    return client


async def old_embeddings(
    slide: Path, roi: Path | None, out: Path, client: JitteryEncoder
) -> Path:
    today = BuilderSettings()
    await build_embeddings(
        slide,
        roi,
        out,
        client=client,  # type: ignore[arg-type]
        num_threads=today.num_threads,
        max_concurrent_tasks=today.max_concurrent_tasks,
        report_progress=lambda _: None,
        **EmbeddingConfiguration().model_dump(),
    )
    return out


async def new_embeddings(
    analysis_id: str, slide: Path, roi: Path | None, out: Path, model: JitteryEncoder
) -> Path:
    analysis = load_catalog(CATALOG).entries("analyses")[analysis_id].value
    inputs = {"slide": slide} | ({"region_of_interest": roi} if roi else {})
    written = await run_analysis(
        analysis,
        inputs,
        {"embeddings": out},
        {ANALYSES[analysis_id]: model},
        concurrency=Concurrency(),
    )
    return written["embeddings"]


@pytest.fixture(scope="module")
def slide(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tissue_slide(tmp_path_factory.mktemp("slide") / "slide.tiff")


@pytest.mark.parametrize("analysis_id", list(ANALYSES))
@pytest.mark.parametrize("with_roi", [False, True], ids=["whole slide", "with roi"])
def test_the_runner_writes_todays_embedding_table_byte_for_byte(
    tmp_path: Path, slide: Path, analysis_id: str, with_roi: bool
) -> None:
    import h5py

    model = encoder(ANALYSES[analysis_id])
    roi = roi_mask(tmp_path / "roi.tiff") if with_roi else None

    old = asyncio.run(old_embeddings(slide, roi, tmp_path / "old.h5", model))
    new = asyncio.run(
        new_embeddings(analysis_id, slide, roi, tmp_path / "new.h5", model)
    )

    with h5py.File(old) as table:
        assert 10 < table.attrs["tile_count"] < 16 * 12
    assert new.read_bytes() == old.read_bytes()


@pytest.mark.parametrize("analysis_id", list(ANALYSES))
def test_a_run_that_skips_every_tile_gives_todays_empty_table(
    tmp_path: Path, slide: Path, analysis_id: str
) -> None:
    import h5py

    nothing = fake_slide(
        tmp_path / "roi.tiff", np.zeros((384, 512, 3), np.uint8), mpp=2.0
    )
    model = encoder(ANALYSES[analysis_id])

    old = asyncio.run(old_embeddings(slide, nothing, tmp_path / "old.h5", model))
    new = asyncio.run(
        new_embeddings(analysis_id, slide, nothing, tmp_path / "new.h5", model)
    )

    with h5py.File(old) as table:
        assert table.attrs["tile_count"] == 0
    assert new.read_bytes() == old.read_bytes()
