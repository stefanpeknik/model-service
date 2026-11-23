"""What the conformance suite needs to run a loader or runtime offline, by name.

A loader usually fetches over a network and a runtime needs weights in its
format, so their options alone can't exercise them. `LOADERS` maps a loader's
name to options that fetch a local weights file; `RUNTIMES` maps a runtime's
name to a function writing tiny weights it loads, with a batch for them.

A loader or runtime with no entry gets only its declaration checked, and its
conformance test is skipped with its name, so the gap shows in the test run.
Add an entry when you add a loader or runtime.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path


@dataclass(frozen=True)
class TinyWeights:
    """Weights a runtime loads in the conformance suite, and a batch for them.

    Attributes:
        weights: The weights file or folder.
        batch: A batch the loaded function takes.
    """

    weights: Path
    batch: np.ndarray


def path_options(weights: Path) -> dict[str, Any]:
    """Options of a loader that takes a local `path`.

    Args:
        weights: A weights file on this machine.

    Returns:
        `{"path": <weights>}`.
    """
    return {"path": str(weights)}


def file_uri_options(weights: Path) -> dict[str, Any]:
    """Options of the `mlflow` loader, through MLflow's local file store.

    Args:
        weights: A weights file on this machine.

    Returns:
        `{"uri": "file://<weights>"}`.
    """
    return {"uri": weights.as_uri()}


def empty_weights(directory: Path) -> TinyWeights:
    """An empty file and two black 8 px uint8 tiles, for a runtime that ignores weights.

    Args:
        directory: An empty directory to write into.

    Returns:
        The file and the batch.
    """
    weights = directory / "weights.bin"
    weights.write_bytes(b"")
    return TinyWeights(weights, np.zeros((2, 3, 8, 8), dtype=np.uint8))


def onnx_graph(directory: Path) -> TinyWeights:
    """An ONNX graph that scores each tile by its mean brightness.

    Args:
        directory: An empty directory to write into.

    Returns:
        The graph and two uint8 8 px tiles.
    """
    from tests.support import onnx_graphs

    graph = onnx_graphs.scores(directory / "scores.onnx")
    return TinyWeights(graph, np.zeros((2, 3, 8, 8), dtype=np.uint8))


def timm_vit(directory: Path) -> TinyWeights:
    """Timm's tiny `test_vit` as a Hub-style folder with random weights.

    Args:
        directory: An empty directory to write into.

    Returns:
        The folder and two float32 tiles of its input size, which the
        runtime's `transform: none` passes through.
    """
    import timm
    from safetensors.torch import save_file

    folder = directory / "test_vit"
    folder.mkdir()
    model = timm.create_model("test_vit", num_classes=0)
    save_file(model.state_dict(), str(folder / "model.safetensors"))
    size = tuple(model.pretrained_cfg["input_size"])  # type: ignore[index]
    (folder / "config.json").write_text(
        json.dumps(
            {
                "architecture": "test_vit",
                "num_classes": 0,
                "pretrained_cfg": {"input_size": list(size)},
            }
        )
    )
    return TinyWeights(folder, np.zeros((2, *size), dtype=np.float32))


LOADERS: dict[str, Callable[[Path], dict[str, Any]]] = {
    "local": path_options,
    "mlflow": file_uri_options,
    # Test components: tests/support/components, the broken ones in
    # test_conformance.py, and the copied template in test_templates.py.
    "local_file": path_options,
    "loses_its_weights": path_options,
    "copied_loader": path_options,
}

RUNTIMES: dict[str, Callable[[Path], TinyWeights]] = {
    "onnx": onnx_graph,
    "timm": timm_vit,
    "constant": empty_weights,
    "loses_a_row": empty_weights,
    "copied_runtime": empty_weights,
}
