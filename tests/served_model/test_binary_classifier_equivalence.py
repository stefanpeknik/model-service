"""`BinaryClassifier` and the generic host give bit-identical results.

Both run the prostate classifier's files (today's model file and the catalog's
served-model file) with only the weights swapped for a stand-in ONNX graph and
the tile shrunk. Deleted with `BinaryClassifier` once prostate passes its
live check ("Served models deploy from catalog/").
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import lz4.frame
import numpy as np
import pytest

from model_service.catalog import load_catalog
from model_service.inference.implementations import binary_classifier
from model_service.served_model import serve
from model_service.served_model.deployment import serve_application
from tests.support import onnx_graphs
from tests.support.model_documents import model_file
from tests.support.off_serve import OffServe, off_serve


if TYPE_CHECKING:
    from collections.abc import Iterator


REPO = Path(__file__).resolve().parents[2]
TILE = 16
BATCH = 8


@pytest.fixture(scope="module")
def graph(tmp_path_factory: pytest.TempPathFactory) -> Path:
    # Black tiles score sigmoid(-12), which prints with an exponent.
    return onnx_graphs.scores(
        tmp_path_factory.mktemp("weights") / "model.onnx", shift=-12
    )


@pytest.fixture(scope="module")
def old(graph: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[OffServe]:
    document = model_file(
        REPO / "deploy/helm/rayservice/models/prostate-classifier-1.yaml"
    )
    config = document["deployment"]["user_config"] | {
        "tile_size": TILE,
        "max_batch_size": BATCH,
        "trt_cache_path": str(tmp_path_factory.mktemp("old-cache")),
        "model": {"_target_": "os.path:abspath", "path": str(graph)},
    }
    with off_serve(
        binary_classifier.BinaryClassifier, binary_classifier.fastapi, config
    ) as running:
        yield running


@pytest.fixture(scope="module")
def new(graph: Path, tmp_path_factory: pytest.TempPathFactory) -> Iterator[OffServe]:
    entry = load_catalog(REPO / "catalog").entries("served-models")[
        "prostate-classifier-1"
    ]
    (deployment,) = serve_application(
        entry, cache_root=str(tmp_path_factory.mktemp("new-cache"))
    )["deployments"]
    config: dict[str, Any] = deployment["user_config"]
    declaration = config["declaration"]
    config["declaration"] = declaration | {
        "weights": {"use": "local", "path": str(graph)},
        "input": declaration["input"] | {"tile_size": TILE},
        "batching": declaration["batching"] | {"max_size": BATCH},
    }
    with off_serve(serve.ServedModelDeployment, serve.fastapi, config) as running:
        yield running


def tiles() -> list[np.ndarray]:
    rng = np.random.default_rng(7)
    batch = list(rng.integers(0, 256, (BATCH - 2, 3, TILE, TILE), dtype=np.uint8))
    return [np.zeros((3, TILE, TILE), np.uint8), *batch, np.full_like(batch[0], 255)]


def test_predict_gives_the_same_floats_for_a_batch(
    old: OffServe, new: OffServe
) -> None:
    batch = tiles()

    expected = old.batch("predict", batch)
    got = new.gather("predict", batch)

    assert all(type(score) is float for score in got)
    assert [s.hex() for s in got] == [s.hex() for s in expected]


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
