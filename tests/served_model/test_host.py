"""The served-model host core through `infer`, `describe` and the raw reply.

Built from a declaration, on stand-in ONNX graphs.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from typing import TYPE_CHECKING, Any, ClassVar

import lz4.frame
import numpy as np
import pytest

from components.base import Runtime
from model_service.catalog.served_models import resolve_served_model
from model_service.registry import Registry, default_registry
from model_service.served_model.guards import ModelInputError, ModelOutputError
from model_service.served_model.host import ServedModelHost
from tests.support import onnx_graphs


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from components.base import BatchFunction, RuntimeContext


TILE = 8


class Recording(Runtime):
    """Test runtime: scores tiles by mean brightness and records batch sizes."""

    name = "recording"
    batches: ClassVar[list[int]] = []

    def load(self, path: Path, context: RuntimeContext) -> BatchFunction:
        def run(tiles: np.ndarray) -> np.ndarray:
            Recording.batches.append(len(tiles))
            return tiles.reshape(len(tiles), -1).mean(axis=1, dtype=np.float32) / 255

        return run


REGISTRY = Registry({(c.kind, c.name): c for c in [*default_registry(), Recording]})


def declaration(weights: Path, **blocks: Any) -> dict[str, Any]:
    return {
        "weights": {"use": "local", "path": str(weights)},
        "runtime": {"use": "onnx"},
        "input": {"tile_size": TILE, "mpp": 0.5},
        "output": {"kind": "scores", "channels": 1, "size": 1, "range": [0, 1]},
        "batching": {"max_size": 4, "wait_s": 0.01},
        "resources": {"memory_gib": 1},
    } | blocks


def host(document: dict[str, Any], cache_root: Path) -> ServedModelHost:
    model = resolve_served_model("m1", document, REGISTRY)
    return ServedModelHost(model, cache_root=cache_root)


def tiles(count: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return list(rng.integers(0, 256, (count, 3, TILE, TILE), dtype=np.uint8))


def sigmoid_scores(batch: list[np.ndarray]) -> np.ndarray:
    means = np.array([t.mean() for t in batch]) / 255
    return 1 / (1 + np.exp(-(8 * means - 4)))


def run(served: ServedModelHost, use: Callable[[ServedModelHost], Any]) -> Any:
    async def scenario() -> Any:
        return await use(served)

    return asyncio.run(scenario())


def test_infer_scores_each_tile_whether_batched_or_alone(tmp_path: Path) -> None:
    # Unusual tensor names: the runtime reads them from the graph.
    graph = onnx_graphs.scores(
        tmp_path / "m.onnx", input_name="pixel_values", output_name="probs"
    )
    batch = tiles(6)

    async def use(served: ServedModelHost) -> tuple[list[Any], list[Any]]:
        together = await asyncio.gather(*(served.infer(t) for t in batch))
        alone = [await served.infer(t) for t in batch]
        return list(together), alone

    together, alone = run(host(declaration(graph), tmp_path), use)

    assert all(s.shape == (1, 1, 1) and s.dtype == np.float32 for s in together)
    np.testing.assert_allclose(np.ravel(together), sigmoid_scores(batch), rtol=1e-6)
    assert [s.tobytes() for s in together] == [s.tobytes() for s in alone]


def test_concurrent_tiles_are_batched_up_to_the_declared_size(
    tmp_path: Path,
) -> None:
    document = declaration(
        tmp_path / "unused",
        runtime={"use": "recording"},
        batching={"max_size": 4, "wait_s": 0.05},
    )
    Recording.batches.clear()

    async def use(served: ServedModelHost) -> None:
        await asyncio.gather(*(served.infer(t) for t in tiles(10)))

    run(host(document, tmp_path), use)

    assert Recording.batches == [4, 4, 2]


def test_describe_names_the_digest_weights_and_active_providers(
    tmp_path: Path,
) -> None:
    graph = onnx_graphs.scores(tmp_path / "m.onnx")
    document = declaration(graph, runtime={"use": "onnx", "tensorrt": True})

    described = host(document, tmp_path / "cache").describe()

    digest = resolve_served_model("m1", document).digest
    assert described == {
        "id": "m1",
        "digest": digest,
        "weights_revision": str(graph),
        # TensorRT and CUDA were asked for; this machine falls back to the CPU.
        "execution_providers": ["CPUExecutionProvider"],
    }
    assert (tmp_path / "cache" / "m1" / digest[:12]).is_dir()


def test_building_logs_what_runs_and_warns_when_tensorrt_fell_back(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    graph = onnx_graphs.scores(tmp_path / "m.onnx")
    document = declaration(graph, runtime={"use": "onnx", "tensorrt": True})

    with caplog.at_level(logging.INFO, logger="model_service"):
        host(document, tmp_path / "cache")

    digest = resolve_served_model("m1", document).digest
    built = [r.getMessage() for r in caplog.records if r.levelno == logging.INFO]
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any(
        digest in m and str(graph) in m and "CPUExecutionProvider" in m for m in built
    ), built
    assert any("TensorrtExecutionProvider" in m for m in warnings), warnings


def test_a_runtime_without_tensorrt_builds_without_a_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    graph = onnx_graphs.scores(tmp_path / "m.onnx")

    with caplog.at_level(logging.INFO, logger="model_service"):
        host(declaration(graph), tmp_path / "cache")

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def raw(served: ServedModelHost, tile: np.ndarray, **headers: str) -> Any:
    body = lz4.frame.compress(np.ascontiguousarray(tile.transpose(1, 2, 0)))
    return run(served, lambda s: s.raw(body, headers))


def test_a_single_score_replies_as_a_json_float(tmp_path: Path) -> None:
    # A black tile scores sigmoid(-12), small enough to print with an exponent.
    graph = onnx_graphs.scores(tmp_path / "m.onnx", shift=-12)
    black, gray = np.zeros((3, TILE, TILE), np.uint8), tiles(1)[0]

    replies = [raw(host(declaration(graph), tmp_path), t) for t in (black, gray)]

    assert [r.media_type for r in replies] == ["application/json"] * 2
    # ONNX Runtime's float32 sigmoid is an approximation, coarse this far out.
    np.testing.assert_allclose(
        [json.loads(r.content) for r in replies],
        [1 / (1 + math.exp(12)), 1 / (1 + math.exp(-(8 * gray.mean() / 255 - 12)))],
        rtol=1e-2,
    )


def test_a_score_map_replies_as_lz4_bytes(tmp_path: Path) -> None:
    graph = onnx_graphs.scores(tmp_path / "m.onnx", per_pixel=True)
    document = declaration(
        graph, output={"kind": "scores", "channels": 1, "size": TILE}
    )
    tile = tiles(1)[0]

    reply = raw(host(document, tmp_path), tile)

    scores = np.frombuffer(lz4.frame.decompress(reply.content), np.float32)
    expected = 1 / (1 + np.exp(-(8 * tile.mean(axis=0) / 255 - 4)))
    assert (reply.media_type, reply.headers) == ("application/octet-stream", {})
    np.testing.assert_allclose(scores.reshape(1, TILE, TILE), [expected], rtol=1e-6)


def test_the_onnx_runtime_casts_its_output_to_the_asked_dtype(
    tmp_path: Path,
) -> None:
    graph = onnx_graphs.scores(tmp_path / "m.onnx", per_pixel=True)
    output = {"kind": "scores", "channels": 1, "size": TILE, "dtype": "float16"}
    document = declaration(
        graph, runtime={"use": "onnx", "output_dtype": "float16"}, output=output
    )
    tile = tiles(1)[0]

    scores = run(host(document, tmp_path), lambda s: s.infer(tile))

    expected = 1 / (1 + np.exp(-(8 * tile.mean(axis=0) / 255 - 4)))
    assert scores.dtype == np.float16
    np.testing.assert_allclose(scores, [expected], rtol=1e-3)


def test_an_embedding_replies_pooled_or_as_tokens_without_registers(
    tmp_path: Path,
) -> None:
    document = declaration(
        onnx_graphs.tokens(tmp_path / "m.onnx"),
        post=[{"use": "cls_mean_pool", "registers": 2}],
        output={"kind": "embedding", "summaries": ["global", "local_mean"]},
    )
    tile = tiles(1)[0]
    tokens = tile.mean(axis=2).T / 255

    pooled = raw(host(document, tmp_path), tile)
    unpooled = raw(
        host(document, tmp_path),
        tile,
        **{"x-pool-tokens": "false", "x-output-dtype": "float16"},
    )

    def features(reply: Any, dtype: type) -> np.ndarray:
        return np.frombuffer(lz4.frame.decompress(reply.content), dtype)

    assert pooled.headers == {"x-output-shape": "(2, 3)"}
    np.testing.assert_allclose(
        features(pooled, np.float32).reshape(2, 3),
        [tokens[0], tokens[3:].mean(axis=0)],
        rtol=1e-6,
    )
    assert unpooled.headers == {"x-output-shape": f"({TILE - 2}, 3)"}
    np.testing.assert_allclose(
        features(unpooled, np.float16).reshape(TILE - 2, 3),
        np.delete(tokens, [1, 2], axis=0),
        rtol=1e-3,
    )


@pytest.mark.parametrize(
    ("graph", "output", "error"),
    [
        (
            {"sigmoid": False, "shift": 2},
            {},
            r"returned \d+\.\d+, outside the declared range \[0\.0, 1\.0\]. "
            r"Hint: a graph that returns logits needs a sigmoid or softmax",
        ),
        (
            {"sigmoid": False, "shift": math.inf},
            {},
            "returned a value that is not finite",
        ),
        ({}, {"size": 2}, r"returned \(1,\) scores per tile, declared \(1, 2, 2\)"),
        ({}, {"dtype": "float16"}, "returned float32, declared float16"),
    ],
    ids=["logits", "infinite", "shape", "dtype"],
)
def test_an_output_that_breaks_its_declaration_fails_the_request(
    tmp_path: Path, graph: dict[str, Any], output: dict[str, Any], error: str
) -> None:
    document = declaration(onnx_graphs.scores(tmp_path / "m.onnx", **graph))
    document["output"] = document["output"] | output

    with pytest.raises(ModelOutputError, match=f"^m1 {error}"):
        run(host(document, tmp_path), lambda s: s.infer(tiles(1)[0]))


@pytest.mark.parametrize(
    "tile",
    [
        np.zeros((3, TILE, TILE), np.float32),
        np.zeros((TILE, TILE, 3), np.uint8),
        [[0]],
    ],
    ids=["float", "channels-last", "list"],
)
def test_a_tile_of_another_dtype_or_shape_is_refused(tmp_path: Path, tile: Any) -> None:
    served = host(declaration(onnx_graphs.scores(tmp_path / "m.onnx")), tmp_path)

    with pytest.raises(ModelInputError, match=r"^m1 takes uint8 RGB tiles \(3, 8, 8\)"):
        run(served, lambda s: s.infer(tile))
