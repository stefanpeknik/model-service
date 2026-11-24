"""The pipeline runner core: progress, ordering, failures, cancellation, outputs.

Each test loads a small pipeline over `fake_slide()` (256 x 256 px at 0.5
µm/px, a pink disc on glass) read as 64 px tiles with stride 32: a 7 x 7 grid.
"""

from __future__ import annotations

import asyncio
import logging
import zlib
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from model_service.catalog import load_catalog
from model_service.contracts.failures import InvalidInvocation, RunFailure
from model_service.formats import score_grid
from model_service.pipeline_runner import Concurrency, run_pipeline
from tests.components.helpers import fake_model, fake_slide, run_component
from tests.support.pipeline_catalog import REGISTRY, model_file, write


if TYPE_CHECKING:
    from pathlib import Path

    from model_service.catalog.pipelines import Pipeline


KEYS = 49
HEATMAP = "[{use: score_grid, output: heatmap}]"
ALL_TILES = "{use: slide_tiles, skip_background: false}"


def pipeline(
    tmp_path: Path,
    sinks: str = HEATMAP,
    *,
    source: str = "{use: slide_tiles}",
    steps: str = "[{model: m1}]",
    model_output: str = "kind: scores",
) -> Pipeline:
    root = tmp_path / "catalog"
    write(root, "models/m1.yaml", model_file(output=model_output))
    write(
        root,
        "pipelines/a1.yaml",
        f"""
        source: {source}
        steps: {steps}
        sinks: {sinks}
        """,
    )
    return load_catalog(root, registry=REGISTRY).pipelines["a1"]


class FakeModel:
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
    resolved: Pipeline,
    slide: Path,
    outputs: dict[str, Path],
    model: FakeModel | None = None,
    **options: Any,
) -> dict[str, Path]:
    models = {"m1": model or FakeModel()} if resolved.models else {}
    return asyncio.run(
        run_pipeline(resolved, {"slide": slide}, outputs, models, **options)
    )


def test_progress_is_the_share_of_keys_done_skipped_tiles_included(
    tmp_path: Path, slide: Path, out: Path
) -> None:
    model = FakeModel()
    progress: list[float] = []

    run(
        pipeline(tmp_path, HEATMAP),
        slide,
        {"heatmap": out / "heatmap.npz"},
        model,
        concurrency=Concurrency(in_flight=1),
        report_progress=progress.append,
    )

    assert 0 < model.calls < KEYS
    assert progress == [i / KEYS for i in range(1, KEYS + 1)]


def test_a_model_free_pipeline_runs_without_handles(
    tmp_path: Path, slide: Path, out: Path
) -> None:
    resolved = pipeline(
        tmp_path,
        source="{use: slide_tiles, tile_size: 64, mpp: 0.5, skip_background: false}",
        steps="[{use: mean_intensity}]",
    )

    written = run(resolved, slide, {"heatmap": out / "heatmap.npz"})

    grid = score_grid.read(written["heatmap"])
    assert grid.values.shape == (1, 8, 8)
    assert grid.scored.all()


@pytest.mark.parametrize(
    ("model", "code", "retryable"),
    [
        (FakeModel(fail_at=5), "model_failed", True),
        (
            FakeModel(output=np.full((1, 1, 1), np.nan, np.float32)),
            "invalid_model_output",
            False,
        ),
        (
            FakeModel(output=np.zeros((2, 1, 1), np.float32)),
            "invalid_model_output",
            False,
        ),
    ],
    ids=["call fails", "not finite", "wrong shape"],
)
def test_a_failing_model_fails_the_run_with_its_code_and_writes_nothing(
    tmp_path: Path, slide: Path, out: Path, model: FakeModel, code: str, retryable: bool
) -> None:
    with pytest.raises(RunFailure) as raised:
        run(pipeline(tmp_path, HEATMAP), slide, {"heatmap": out / "heatmap.npz"}, model)

    assert (raised.value.code, raised.value.retryable) == (code, retryable)
    assert list(out.iterdir()) == []


def test_features_that_change_shape_between_tiles_fail_the_embedding_table(
    tmp_path: Path, slide: Path, out: Path
) -> None:
    resolved = pipeline(
        tmp_path,
        "[{use: embedding_table, output: embeddings}]",
        source=ALL_TILES,
        model_output="kind: embedding, summaries: [tile]",
    )

    class Growing(FakeModel):
        async def infer(self, tile: np.ndarray) -> np.ndarray:
            self.calls += 1
            return np.ones(4 if self.calls < 5 else 5, np.float32)

    with pytest.raises(RunFailure) as raised:
        run(resolved, slide, {"embeddings": out / "embeddings.h5"}, Growing())

    assert (raised.value.code, raised.value.retryable) == (
        "invalid_model_output",
        False,
    )
    assert list(out.iterdir()) == []


def test_cancelling_a_run_leaves_no_output(
    tmp_path: Path, slide: Path, out: Path
) -> None:
    resolved = pipeline(tmp_path, HEATMAP)
    model = FakeModel(stall_at=3)

    async def cancel_when_stalled() -> None:
        task = asyncio.create_task(
            run_pipeline(
                resolved, {"slide": slide}, {"heatmap": out / "h.npz"}, {"m1": model}
            )
        )
        await model.stalled.wait()
        task.cancel()
        await task

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(cancel_when_stalled())
    assert list(out.iterdir()) == []


def test_cancelling_a_run_waits_for_running_reads_before_closing_the_source(
    tmp_path: Path, slide: Path, out: Path
) -> None:
    resolved = pipeline(
        tmp_path,
        "[{use: score_table, output: table}]",
        source="{use: slow_tiles, load_s: 0.5}",
        steps="[{use: mean_intensity}]",
    )

    async def cancel_while_reading() -> None:
        task = asyncio.create_task(
            run_pipeline(resolved, {"slide": slide}, {"table": out / "t.npz"}, {})
        )
        await asyncio.sleep(0.1)
        task.cancel()
        await task

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(cancel_while_reading())
    assert list(out.iterdir()) == []


def test_several_sinks_write_all_their_outputs_or_none(
    tmp_path: Path, slide: Path, out: Path
) -> None:
    outputs = {"heatmap": out / "heatmap.npz", "table": out / "table.npz"}
    two_sinks = (
        "[{use: score_grid, output: heatmap}, {use: score_table, output: table%s}]"
    )
    run(pipeline(tmp_path, two_sinks % ""), slide, outputs)
    before = {name: path.read_bytes() for name, path in outputs.items()}

    # The failing run scores differently, so a heatmap it replaced would show.
    other_scores = FakeModel(output=np.full((1, 1, 1), 0.25, np.float32))
    with pytest.raises(RunFailure, match="failed unexpectedly") as raised:
        run(
            pipeline(tmp_path, two_sinks % ", fail_on_close: true"),
            slide,
            outputs,
            other_scores,
        )

    assert (raised.value.code, raised.value.retryable) == ("run_internal_error", True)
    assert {name: path.read_bytes() for name, path in outputs.items()} == before
    assert sorted(p.name for p in out.iterdir()) == ["heatmap.npz", "table.npz"]


def test_an_ordered_sink_gets_rows_in_key_order(
    tmp_path: Path, slide: Path, out: Path
) -> None:
    resolved = pipeline(
        tmp_path,
        "[{use: key_log, output: keys}, {use: score_grid, output: heatmap}]",
        source=ALL_TILES,
    )
    keys = run_component(resolved.source, work_dir=tmp_path).keys

    written = run(
        resolved,
        slide,
        {"keys": out / "keys.npy", "heatmap": out / "heatmap.npz"},
        FakeModel(jitter=True),
        concurrency=Concurrency(in_flight=8),
    )

    assert [tuple(row) for row in np.load(written["keys"]).tolist()] == [
        (int(x), int(y)) for x, y in keys
    ]


def test_the_run_logs_the_time_each_step_takes(
    tmp_path: Path, slide: Path, out: Path, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.INFO, logger="model_service.pipeline_runner"):
        run(pipeline(tmp_path, HEATMAP), slide, {"heatmap": out / "heatmap.npz"})

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
        ({"slide": "slide.tiff"}, {"heatmap": "out"}, "is a directory"),
        ({"slide": "slide.tiff"}, {"heatmap": "slide.tiff/h.npz"}, "not a directory"),
        ({"slide": "slide.tiff"}, {"heatmap": "x" * 250 + ".npz"}, "name is too long"),
    ],
)
def test_inputs_and_outputs_that_do_not_fit_are_refused(
    tmp_path: Path,
    slide: Path,
    out: Path,
    inputs: dict[str, str],
    outputs: dict[str, str],
    message: str,
) -> None:
    resolved = pipeline(tmp_path, HEATMAP)

    with pytest.raises(InvalidInvocation) as raised:
        asyncio.run(
            run_pipeline(
                resolved,
                {k: tmp_path / v for k, v in inputs.items()},
                {k: tmp_path / v for k, v in outputs.items()},
                {"m1": FakeModel()},
            )
        )

    assert message in str(raised.value)


def test_a_directory_given_as_an_output_is_refused_and_left_intact(
    tmp_path: Path, slide: Path, out: Path
) -> None:
    (out / "keep.txt").write_text("results")

    with pytest.raises(InvalidInvocation, match="is a directory"):
        run(pipeline(tmp_path, HEATMAP), slide, {"heatmap": out})

    assert (out / "keep.txt").read_text() == "results"


def test_two_outputs_with_the_same_destination_are_refused(
    tmp_path: Path, slide: Path, out: Path
) -> None:
    two_sinks = (
        "[{use: score_grid, output: heatmap}, {use: score_table, output: table}]"
    )
    same = {"heatmap": out / "a.npz", "table": out / "sub" / ".." / "a.npz"}

    with pytest.raises(InvalidInvocation, match="same path"):
        run(pipeline(tmp_path, two_sinks), slide, same)

    assert list(out.iterdir()) == []
