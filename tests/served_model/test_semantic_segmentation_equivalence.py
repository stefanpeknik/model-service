"""`SemanticSegmentation` and the generic host give bit-identical results.

Both run EPISeg's files (today's model file and the catalog's served-model
file) with only the weights swapped for a stand-in ONNX graph and the tile
and score map shrunk. The sessions both build are recorded where they meet
ONNX Runtime. Deleted with `SemanticSegmentation` once EPISeg passes
its live check ("EPISeg on the generic host").
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lz4.frame
import numpy as np
import onnxruntime as ort
import pytest

from model_service.catalog import load_catalog
from model_service.inference.implementations import semantic_segmentation
from model_service.served_model import serve
from model_service.served_model.deployment import serve_application
from tests.support import onnx_graphs
from tests.support.model_documents import model_file
from tests.support.off_serve import off_serve


if TYPE_CHECKING:
    from collections.abc import Iterator

    from tests.support.off_serve import OffServe


REPO = Path(__file__).resolve().parents[2]
TILE = 16
BATCH = 8


@pytest.fixture(scope="module")
def graph(tmp_path_factory: pytest.TempPathFactory) -> Path:
    # A score per pixel, so the map has the tile's size like EPISeg's.
    return onnx_graphs.scores(
        tmp_path_factory.mktemp("weights") / "model.onnx", per_pixel=True
    )


@contextmanager
def recording_sessions() -> Iterator[list[ort.InferenceSession]]:
    """Record every ONNX Runtime session created inside the block, in order."""
    created: list[ort.InferenceSession] = []

    class Recorded(ort.InferenceSession):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            created.append(self)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(ort, "InferenceSession", Recorded)
        yield created


@pytest.fixture(scope="module")
def sessions() -> dict[str, list[ort.InferenceSession]]:
    return {}


@pytest.fixture(scope="module")
def old(
    graph: Path,
    sessions: dict[str, list[ort.InferenceSession]],
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[OffServe]:
    document = model_file(REPO / "deploy/helm/rayservice/models/episeg-1.yaml")
    config = document["deployment"]["user_config"] | {
        "tile_size": TILE,
        "output_tile_size": TILE,
        "max_batch_size": BATCH,
        "trt_cache_path": str(tmp_path_factory.mktemp("old-cache")),
        "model": {"_target_": "os.path:abspath", "path": str(graph)},
    }
    with (
        recording_sessions() as sessions["old"],
        off_serve(
            semantic_segmentation.SemanticSegmentation,
            semantic_segmentation.fastapi,
            config,
        ) as running,
    ):
        yield running


@pytest.fixture(scope="module")
def new(
    graph: Path,
    sessions: dict[str, list[ort.InferenceSession]],
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[OffServe]:
    entry = load_catalog(REPO / "catalog").entries("served-models")["episeg-1"]
    (deployment,) = serve_application(
        entry, cache_root=str(tmp_path_factory.mktemp("new-cache"))
    )["deployments"]
    config: dict[str, Any] = deployment["user_config"]
    declaration = config["declaration"]
    config["declaration"] = declaration | {
        "weights": {"use": "local", "path": str(graph)},
        "input": declaration["input"] | {"tile_size": TILE},
        "output": declaration["output"] | {"size": TILE},
        "batching": declaration["batching"] | {"max_size": BATCH},
    }
    with (
        recording_sessions() as sessions["new"],
        off_serve(serve.ServedModelDeployment, serve.fastapi, config) as running,
    ):
        yield running


def tiles() -> list[np.ndarray]:
    rng = np.random.default_rng(7)
    batch = list(rng.integers(0, 256, (BATCH - 2, 3, TILE, TILE), dtype=np.uint8))
    return [np.zeros((3, TILE, TILE), np.uint8), *batch, np.full_like(batch[0], 255)]


def test_predict_gives_the_same_float16_maps_for_a_batch(
    old: OffServe, new: OffServe
) -> None:
    batch = tiles()

    expected = old.batch("predict", batch)
    got = new.gather("predict", batch)

    assert [(m.dtype, m.shape) for m in got] == [(m.dtype, m.shape) for m in expected]
    assert [m.tobytes() for m in got] == [m.tobytes() for m in expected]


def test_the_raw_route_replies_with_the_same_bytes(
    old: OffServe, new: OffServe
) -> None:
    for tile in tiles():
        body = lz4.frame.compress(np.ascontiguousarray(tile.transpose(1, 2, 0)))

        expected, got = old.post(body), new.post(body)

        assert (got.status_code, got.headers["content-type"], got.content) == (
            expected.status_code,
            expected.headers["content-type"],
            expected.content,
        )


def test_get_config_reports_the_same_fields(old: OffServe, new: OffServe) -> None:
    assert new.call("get_config") == old.call("get_config")


def test_the_session_keeps_onnx_runtime_defaults_as_today(
    old: OffServe, new: OffServe, sessions: dict[str, list[ort.InferenceSession]]
) -> None:
    def settings(options: ort.SessionOptions) -> tuple[Any, ...]:
        return (
            options.intra_op_num_threads,
            options.inter_op_num_threads,
            options.graph_optimization_level,
            options.execution_mode,
        )

    # The last session is the one that infers; the host may probe the graph first.
    got, expected = (
        sessions[side][-1].get_session_options() for side in ("new", "old")
    )

    # SemanticSegmentation hands its options to a keyword ONNX Runtime ignores.
    assert settings(expected) == settings(ort.SessionOptions())
    assert settings(got) == settings(expected)
