"""The analysis runner core: progress, ordering, failures, cancellation, outputs.

Each test loads a small analysis over `fake_slide()` (256 x 256 px at 0.5
µm/px, a pink disc on glass) read as 64 px tiles with stride 32: a 7 x 7 grid.
"""

from __future__ import annotations

import asyncio
import logging
import zlib
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from model_service.analysis_runner import Concurrency, run_analysis
from model_service.catalog import load_catalog
from model_service.contracts.failures import BuilderFailure, InvalidInvocation
from model_service.formats import score_grid
from model_service.testing import fake_model, fake_slide, run_component
from tests.support.analysis_catalog import REGISTRY, served_model, write


if TYPE_CHECKING:
    from pathlib import Path

    from model_service.catalog.analyses import Analysis


KEYS = 49
HEATMAP = "[{use: score_grid, output: heatmap}]"
ALL_TILES = "{use: slide_tiles, skip_background: false}"


def analysis(
    tmp_path: Path,
    sinks: str = HEATMAP,
    *,
    source: str = "{use: slide_tiles}",
    steps: str = "[{model: m1}]",
    model_output: str = "kind: scores",
) -> Analysis:
    root = tmp_path / "catalog"
    write(root, "served-models/m1.yaml", served_model(output=model_output))
    write(
        root,
        "analyses/a1.yaml",
        f"""
        source: {source}
        steps: {steps}
        sinks: {sinks}
        """,
    )
    return load_catalog(root, registry=REGISTRY).entries("analyses")["a1"].value


class Model:
    """The fake model, counting calls; it can fail, stall or answer late."""

    def __init__(
        self,
        *,
        fail_at: int = 0,
        stall_at: int = 0,
        output: Any = None,
        jitter: bool = False,
    ) -> None:
        self.model = fake_model(tile_size=64, mpp=0.5)
        self.calls = 0
        self.fail_at, self.stall_at, self.output, self.jitter = (
            fail_at,
            stall_at,
            output,
            jitter,
        )
        self.stalled = asyncio.Event()

    async def infer(self, tile: np.ndarray) -> np.ndarray:
        self.calls += 1
        if self.calls == self.fail_at:
            raise ConnectionError("replica died")
        if self.calls == self.stall_at:
            self.stalled.set()
            await asyncio.Event().wait()
        if self.jitter:
            await asyncio.sleep(zlib.crc32(tile.tobytes()) % 5 / 1000)
        return self.output if self.output is not None else await self.model.infer(tile)


@pytest.fixture
def slide(tmp_path: Path) -> Path:
    return fake_slide(tmp_path / "slide.tiff")


@pytest.fixture
def out(tmp_path: Path) -> Path:
    directory = tmp_path / "out"
    directory.mkdir()
    return directory


def run(
    resolved: Analysis,
    slide: Path,
    outputs: dict[str, Path],
    model: Model | None = None,
    **options: Any,
) -> dict[str, Path]:
    models = {"m1": model or Model()} if resolved.models else {}
    return asyncio.run(
        run_analysis(resolved, {"slide": slide}, outputs, models, **options)
    )


def test_progress_is_the_share_of_keys_done_skipped_tiles_included(
    tmp_path: Path, slide: Path, out: Path
) -> None:
    model = Model()
    progress: list[float] = []

    run(
        analysis(tmp_path, HEATMAP),
        slide,
        {"heatmap": out / "heatmap.npz"},
        model,
        concurrency=Concurrency(in_flight=1),
        report_progress=progress.append,
    )

    assert 0 < model.calls < KEYS
    assert progress == [i / KEYS for i in range(1, KEYS + 1)]


def test_a_model_free_analysis_runs_without_handles(
    tmp_path: Path, slide: Path, out: Path
) -> None:
    resolved = analysis(
        tmp_path,
        source="{use: slide_tiles, tile_size: 64, mpp: 0.5, skip_background: false}",
        steps="[{use: mean_intensity}]",
    )

    written = run(resolved, slide, {"heatmap": out / "heatmap.npz"})

    grid = score_grid.read(written["heatmap"])
    assert score_grid.check(written["heatmap"]) == []
    assert grid.values.shape == (1, 8, 8)
    assert grid.scored.all()


@pytest.mark.parametrize(
    ("model", "code", "retryable"),
    [
        (Model(fail_at=5), "model_failed", True),
        (
            Model(output=np.full((1, 1, 1), np.nan, np.float32)),
            "invalid_model_output",
            False,
        ),
        (Model(output=np.zeros((2, 1, 1), np.float32)), "invalid_model_output", False),
    ],
    ids=["call fails", "not finite", "wrong shape"],
)
def test_a_failing_model_fails_the_run_with_its_code_and_writes_nothing(
    tmp_path: Path, slide: Path, out: Path, model: Model, code: str, retryable: bool
) -> None:
    with pytest.raises(BuilderFailure) as raised:
        run(analysis(tmp_path, HEATMAP), slide, {"heatmap": out / "heatmap.npz"}, model)

    assert (raised.value.code, raised.value.retryable) == (code, retryable)
    assert list(out.iterdir()) == []


def test_features_that_change_shape_between_tiles_fail_the_embedding_table(
    tmp_path: Path, slide: Path, out: Path
) -> None:
    resolved = analysis(
        tmp_path,
        "[{use: embedding_table, output: embeddings}]",
        source=ALL_TILES,
        model_output="kind: embedding, summaries: [tile]",
    )

    class Growing(Model):
        async def infer(self, tile: np.ndarray) -> np.ndarray:
            self.calls += 1
            return np.ones(4 if self.calls < 5 else 5, np.float32)

    with pytest.raises(BuilderFailure) as raised:
        run(resolved, slide, {"embeddings": out / "embeddings.h5"}, Growing())

    assert (raised.value.code, raised.value.retryable) == (
        "invalid_model_output",
        False,
    )
    assert list(out.iterdir()) == []


def test_cancelling_a_run_leaves_no_output(
    tmp_path: Path, slide: Path, out: Path
) -> None:
    resolved = analysis(tmp_path, HEATMAP)
    model = Model(stall_at=3)

    async def cancel_when_stalled() -> None:
        task = asyncio.create_task(
            run_analysis(
                resolved, {"slide": slide}, {"heatmap": out / "h.npz"}, {"m1": model}
            )
        )
        await model.stalled.wait()
        task.cancel()
        await task

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(cancel_when_stalled())
    assert list(out.iterdir()) == []


def test_several_sinks_write_all_their_outputs_or_none(
    tmp_path: Path, slide: Path, out: Path
) -> None:
    outputs = {"heatmap": out / "heatmap.npz", "table": out / "table.npz"}
    two_sinks = (
        "[{use: score_grid, output: heatmap}, {use: score_table, output: table%s}]"
    )
    run(analysis(tmp_path, two_sinks % ""), slide, outputs)
    before = {name: path.read_bytes() for name, path in outputs.items()}

    # The failing run scores differently, so a heatmap it replaced would show.
    other_scores = Model(output=np.full((1, 1, 1), 0.25, np.float32))
    with pytest.raises(BuilderFailure, match="failed unexpectedly"):
        run(
            analysis(tmp_path, two_sinks % ", fail_on_close: true"),
            slide,
            outputs,
            other_scores,
        )

    assert {name: path.read_bytes() for name, path in outputs.items()} == before
    assert sorted(p.name for p in out.iterdir()) == ["heatmap.npz", "table.npz"]


def test_an_ordered_sink_gets_rows_in_key_order(
    tmp_path: Path, slide: Path, out: Path
) -> None:
    resolved = analysis(
        tmp_path,
        "[{use: key_log, output: keys}, {use: score_grid, output: heatmap}]",
        source=ALL_TILES,
    )
    keys = run_component(resolved.source, work_dir=tmp_path).keys

    written = run(
        resolved,
        slide,
        {"keys": out / "keys.npy", "heatmap": out / "heatmap.npz"},
        Model(jitter=True),
        concurrency=Concurrency(in_flight=8),
    )

    assert [tuple(row) for row in np.load(written["keys"]).tolist()] == [
        (int(x), int(y)) for x, y in keys
    ]


def test_the_run_logs_the_time_each_step_takes(
    tmp_path: Path, slide: Path, out: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="model_service.analysis_runner"):
        run(analysis(tmp_path, HEATMAP), slide, {"heatmap": out / "heatmap.npz"})

    summary = caplog.records[-1].getMessage()
    assert f"{KEYS} of {KEYS} keys" in summary
    for stage in ("slide_tiles", "model m1", "score_grid"):
        assert f"{stage} " in summary and " ms" in summary


@pytest.mark.parametrize(
    ("inputs", "outputs", "message"),
    [
        ({}, {"heatmap": "h.npz"}, "missing required ['slide']"),
        (
            {"slide": "slide.tiff", "mask": "slide.tiff"},
            {"heatmap": "h.npz"},
            "unknown ['mask']",
        ),
        ({"slide": "slide.tiff"}, {}, "missing ['heatmap']"),
        ({"slide": "slide.tiff"}, {"heatmap": "h.npz", "x": "x.npz"}, "unknown ['x']"),
        (
            {"slide": "slide.tiff"},
            {"heatmap": "slide.tiff"},
            "must not replace an input",
        ),
    ],
)
def test_inputs_and_outputs_that_do_not_fit_are_refused(
    tmp_path: Path,
    slide: Path,
    inputs: dict[str, str],
    outputs: dict[str, str],
    message: str,
) -> None:
    resolved = analysis(tmp_path, HEATMAP)

    with pytest.raises(InvalidInvocation) as raised:
        asyncio.run(
            run_analysis(
                resolved,
                {k: tmp_path / v for k, v in inputs.items()},
                {k: tmp_path / v for k, v in outputs.items()},
                {"m1": Model()},
            )
        )

    assert message in str(raised.value)
