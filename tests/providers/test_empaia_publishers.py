"""Publishers carry stored outputs into EMPAIA, and finish publishes all or nothing.

The equivalence test replays the score grid of the recorded fixture through
`continuous_pixelmap`; the fixture holds the pixelmap the prostate app
uploaded for it, recorded at e91f62d.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
from rationai.empaia.typing import SlideInfo

from components import EmpaiaOutput
from model_service.contracts.result import RunResult, SharedArtifact
from model_service.control_plane.scheduler.public import (
    ProviderFinishFailureError,
    ProviderSuccess,
)
from model_service.control_plane.scheduler.targets import ExecutionTarget
from model_service.formats import score_grid, value
from model_service.providers.empaia.app_service.gateway import EmpaiaGatewayError
from model_service.providers.empaia.jobs.adapter import EmpaiaProviderAdapter
from model_service.registry.empaia import publisher
from tests.support.empaia_fixtures import SLIDE_ID, slide_info


FIXTURES = Path(__file__).parent / "fixtures"
HEATMAP = EmpaiaOutput("heatmap", "slide")
CONTEXT = {"api_url": "https://empaia.invalid", "job_id": "j", "credential": "c"}


class PyramidGateway:
    """Serves one slide and records each pixelmap and each posted body."""

    job_id = "job-1"

    def __init__(self, slide: SlideInfo | None = None) -> None:
        self.slide = slide or slide_info()
        self.published: list[tuple[str, str, tuple[Any, ...]]] = []
        self.posted: list[tuple[str, Any]] = []
        self.slides_read: list[str] = []

    async def slide_info(self, name: str) -> SlideInfo:
        self.slides_read.append(name)
        return self.slide

    async def publish_pixelmap(
        self, name: str, slide_id: str, tile_sets: tuple[Any, ...]
    ) -> None:
        self.published.append((name, slide_id, tile_sets))

    async def post_output(self, name: str, body: Any) -> None:
        self.posted.append((name, body))


def grid(values: np.ndarray, scored: np.ndarray) -> score_grid.ScoreGrid:
    # 64 px per score at level 0: 4 px at level 4, 2 px at level 5.
    return score_grid.ScoreGrid(
        values=values,
        scored=scored,
        level=0,
        level_size=(384, 256),
        cell_size=64.0,
        mpp=(0.25, 0.25),
    )


async def publish_grid(
    path: Path,
    gateway: PyramidGateway,
    media_type: str = score_grid.FORMAT.media_type,
    tile_size: int = 256,
) -> None:
    pixelmap = publisher(
        {"type": "continuous_pixelmap", "tile_size": tile_size}, "score_grid"
    )
    commit = await pixelmap.prepare(gateway, HEATMAP, path, media_type)
    assert gateway.published == []
    await commit()


class ContinuousPixelmapTests(unittest.IsolatedAsyncioTestCase):
    async def test_publisher_calls_match_the_recording_for_the_same_score_grid(
        self,
    ) -> None:
        before = np.load(FIXTURES / "prostate-heatmap-pixelmap-before.npz")
        gateway = PyramidGateway(slide_info(2048, 1536, 7))

        await publish_grid(FIXTURES / "prostate-heatmap-grid.npz", gateway)

        [(name, slide_id, tile_sets)] = gateway.published
        assert (name, slide_id) == (str(before["name"]), str(before["slide_id"]))
        assert len(tile_sets) == sum(1 for k in before.files if k.endswith("_header"))
        for i, tile_set in enumerate(tile_sets):
            header = [
                tile_set.level,
                tile_set.tile_width,
                tile_set.tile_height,
                tile_set.min_tile_x,
                tile_set.min_tile_y,
                tile_set.max_tile_x,
                tile_set.max_tile_y,
            ]
            assert header == before[f"set{i}_header"].tolist()
            positions = [[t.x, t.y] for t in tile_set.tiles]
            assert positions == before[f"set{i}_positions"].tolist()
            np.testing.assert_array_equal(
                np.stack([np.asarray(t.data) for t in tile_set.tiles]),
                before[f"set{i}_data"],
            )

    async def test_scores_are_published_exactly_from_the_coarsest_level_keeping_four_pixels_per_score(
        self,
    ) -> None:
        values = np.zeros((1, 4, 6), dtype=np.float32)
        scored = np.zeros((4, 6), dtype=bool)
        values[0, 1, 2], scored[1, 2] = 0.75, True
        values[0, 3, 5], scored[3, 5] = 0.25, True
        gateway = PyramidGateway()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "heatmap.npz"
            score_grid.write(path, grid(values, scored))
            await publish_grid(path, gateway)

        [(name, slide_id, tile_sets)] = gateway.published
        assert (name, slide_id, gateway.slides_read) == ("heatmap", SLIDE_ID, ["slide"])
        assert [tile_set.level for tile_set in tile_sets] == [4, 5]
        [tile] = tile_sets[0].tiles
        data = np.asarray(tile.data)
        assert data[4:8, 8:12].tolist() == [[0.75] * 4] * 4
        assert data[12:16, 20:24].tolist() == [[0.25] * 4] * 4
        # Cells the model never scored are 0 in the tile, not scores.
        assert float(data[0:4, 0:4].max()) == 0.0
        assert float(data.max()) == 0.75

    async def test_grids_it_cannot_publish_are_invalid_before_any_write(self) -> None:
        nothing_scored = grid(np.zeros((1, 4, 6), np.float32), np.zeros((4, 6), bool))
        two_channels = grid(np.zeros((2, 4, 6), np.float32), np.ones((4, 6), bool))
        with TemporaryDirectory() as directory:
            root = Path(directory)
            score_grid.write(root / "nothing.npz", nothing_scored)
            score_grid.write(root / "two.npz", two_channels)
            (root / "unreadable.npz").write_bytes(b"not an npz")
            cases = {
                "nothing scored": (root / "nothing.npz", score_grid.FORMAT.media_type),
                "two channels": (root / "two.npz", score_grid.FORMAT.media_type),
                "unreadable": (root / "unreadable.npz", score_grid.FORMAT.media_type),
                "a version it doesn't know": (
                    root / "two.npz",
                    "application/vnd.rationai.score-grid+npz; version=9",
                ),
            }
            for case, (path, media_type) in cases.items():
                gateway = PyramidGateway()
                with (
                    self.subTest(case),
                    self.assertRaises(EmpaiaGatewayError) as raised,
                ):
                    await publish_grid(path, gateway, media_type)
                assert not raised.exception.retryable
                assert not raised.exception.outcome_unknown
                assert gateway.published == []


class PrimitiveTests(unittest.IsolatedAsyncioTestCase):
    async def test_a_value_is_posted_as_the_named_primitive_created_by_the_job(
        self,
    ) -> None:
        gateway = PyramidGateway()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "score.json"
            value.write(path, 1)
            primitive = publisher({"type": "float"}, "value")
            commit = await primitive.prepare(
                gateway, EmpaiaOutput("score"), path, value.FORMAT.media_type
            )
            assert gateway.posted == []
            await commit()

        [(name, body)] = gateway.posted
        assert name == "score"
        assert {k: v for k, v in body.model_dump().items() if v is not None} == {
            "name": "score",
            "type": "float",
            "value": 1,
            "creator_id": "job-1",
            "creator_type": "job",
        }

    async def test_a_value_of_another_type_is_rejected_before_any_write(self) -> None:
        for output_type, scalar in (
            ("integer", True),
            ("integer", 0.5),
            ("float", False),
            ("bool", 1),
            ("string", ""),
        ):
            gateway = PyramidGateway()
            with (
                self.subTest(output_type=output_type, value=scalar),
                TemporaryDirectory() as directory,
            ):
                path = Path(directory) / "score.json"
                value.write(path, scalar)
                with self.assertRaises(EmpaiaGatewayError) as raised:
                    await publisher({"type": output_type}, "value").prepare(
                        gateway, EmpaiaOutput("score"), path, value.FORMAT.media_type
                    )
                assert not raised.exception.retryable
                assert gateway.posted == []


def analysis_success(
    outputs: dict[str, SharedArtifact], sinks: list[str]
) -> ProviderSuccess:
    return ProviderSuccess(
        RunResult(outputs=dict(outputs)),
        ExecutionTarget(
            "a1",
            ("m1",),
            "digest",
            {"sinks": [{"use": "x", "output": name} for name in sinks]},
        ),
    )


def admitted(**outputs: dict[str, Any]) -> dict[str, Any]:
    """A protected context with the publications admission recorded."""
    return {**CONTEXT, "publications": {"slide": "slide", "outputs": outputs}}


def heatmap_file(directory: str, name: str = "heatmap.npz") -> Path:
    path = Path(directory) / name
    values = np.full((1, 4, 6), 0.5, np.float32)
    score_grid.write(path, grid(values, np.ones((4, 6), bool)))
    return path


class FinishTests(unittest.IsolatedAsyncioTestCase):
    def provider(self, gateway: Any) -> EmpaiaProviderAdapter:
        gateway.set_before_mutation = lambda _callback: None
        return EmpaiaProviderAdapter(
            lambda _context: gateway, allowed_api_urls=("https://empaia.invalid",)
        )

    async def test_an_analysis_publishes_the_outputs_it_was_admitted_with(
        self,
    ) -> None:
        gateway = AsyncMock()
        gateway.slide_info.return_value = slide_info()
        with TemporaryDirectory() as directory:
            await self.provider(gateway).finish(
                admitted(heatmap={"type": "continuous_pixelmap", "tile_size": 512}),
                analysis_success(
                    {
                        "heatmap": SharedArtifact(
                            str(heatmap_file(directory)), score_grid.FORMAT.media_type
                        ),
                        # Not listed under `outputs`, so not published.
                        "other": SharedArtifact(
                            str(heatmap_file(directory, "other.npz")),
                            score_grid.FORMAT.media_type,
                        ),
                    },
                    ["heatmap", "other"],
                ),
            )

        [call] = gateway.publish_pixelmap.await_args_list
        assert call.args[:2] == ("heatmap", SLIDE_ID)
        gateway.slide_info.assert_awaited_once_with("slide")
        gateway.finalize.assert_awaited_once()

    async def test_a_result_unlike_the_analysis_is_never_published(self) -> None:
        gateway = AsyncMock()

        with self.assertRaises(ProviderFinishFailureError) as raised:
            await self.provider(gateway).finish(
                admitted(heatmap={"type": "continuous_pixelmap"}),
                analysis_success(
                    {
                        "extra": SharedArtifact(
                            "/shared/x.npz", score_grid.FORMAT.media_type
                        )
                    },
                    ["heatmap"],
                ),
            )

        assert not raised.exception.retryable
        assert not raised.exception.outcome_unknown
        gateway.finalize.assert_not_awaited()

    async def test_an_unreadable_artifact_is_invalid_not_retried(self) -> None:
        gateway = AsyncMock()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "heatmap.npz"
            path.write_bytes(b"truncated")

            with self.assertRaises(ProviderFinishFailureError) as raised:
                await self.provider(gateway).finish(
                    admitted(heatmap={"type": "continuous_pixelmap"}),
                    analysis_success(
                        {
                            "heatmap": SharedArtifact(
                                str(path), score_grid.FORMAT.media_type
                            )
                        },
                        ["heatmap"],
                    ),
                )

        assert not raised.exception.retryable
        assert not raised.exception.outcome_unknown

    async def test_an_invalid_output_stops_the_job_before_any_write(self) -> None:
        gateway = AsyncMock()
        gateway.slide_info.return_value = slide_info()
        with TemporaryDirectory() as directory:
            score = Path(directory) / "score.json"
            value.write(score, "high")

            with self.assertRaises(ProviderFinishFailureError) as raised:
                await self.provider(gateway).finish(
                    admitted(
                        heatmap={"type": "continuous_pixelmap"},
                        score={"type": "float"},
                    ),
                    analysis_success(
                        {
                            "heatmap": SharedArtifact(
                                str(heatmap_file(directory)),
                                score_grid.FORMAT.media_type,
                            ),
                            "score": SharedArtifact(
                                str(score), value.FORMAT.media_type
                            ),
                        },
                        ["heatmap", "score"],
                    ),
                )

        # The heatmap was prepared, but its write never ran.
        gateway.slide_info.assert_awaited_once()
        gateway.publish_pixelmap.assert_not_awaited()
        gateway.post_output.assert_not_awaited()
        assert not raised.exception.outcome_unknown


if __name__ == "__main__":
    unittest.main()
