"""Tiny ONNX graphs standing in for real weights, built with `onnx.helper`.

Every graph takes uint8 RGB tiles `(N, 3, T, T)` like the real ONNX models.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


if TYPE_CHECKING:
    from pathlib import Path


OPSET = 17
IR_VERSION = 8


def scores(
    path: Path,
    *,
    scale: float = 8.0,
    shift: float = -4.0,
    sigmoid: bool = True,
    per_pixel: bool = False,
    input_name: str = "input",
    output_name: str = "output",
) -> Path:
    """A classifier: `sigmoid(scale * mean / 255 + shift)` per tile, `(N, 1)`.

    Args:
        path: Where to write the graph.
        scale: Multiplies the mean brightness in [0, 1].
        shift: Added after scaling; `inf` makes every score infinite.
        sigmoid: False returns the logits.
        per_pixel: Average over channels only, giving an `(N, 1, T, T)` map.
        input_name: The graph input's name.
        output_name: The graph output's name.

    Returns:
        The path.
    """
    axes = [1] if per_pixel else [1, 2, 3]
    nodes = [
        helper.make_node("Cast", [input_name], ["x"], to=TensorProto.FLOAT),
        helper.make_node("ReduceMean", ["x"], ["mean"], axes=axes, keepdims=1),
        helper.make_node("Div", ["mean", "c255"], ["unit"]),
        helper.make_node("Mul", ["unit", "scale"], ["scaled"]),
        helper.make_node("Add", ["scaled", "shift"], ["_logits"]),
    ]
    last = "_logits"
    if sigmoid:
        nodes.append(helper.make_node("Sigmoid", [last], ["_probs"]))
        last = "_probs"
    if per_pixel:
        nodes.append(helper.make_node("Identity", [last], [output_name]))
        out_shape = ["N", 1, "T", "T"]
    else:
        nodes.append(helper.make_node("Reshape", [last, "flat"], [output_name]))
        out_shape = ["N", 1]
    constants = {
        "c255": np.float32(255),
        "scale": np.float32(scale),
        "shift": np.float32(shift),
        "flat": np.array([-1, 1], dtype=np.int64),
    }
    return _save(path, nodes, constants, input_name, output_name, out_shape)


def tokens(path: Path) -> Path:
    """An encoder: `T` tokens of dimension 3 per tile, `(N, T, 3)`.

    Token `i` is the mean of pixel row `i` per channel, scaled to [0, 1].

    Args:
        path: Where to write the graph.

    Returns:
        The path.
    """
    nodes = [
        helper.make_node("Cast", ["input"], ["x"], to=TensorProto.FLOAT),
        helper.make_node("ReduceMean", ["x"], ["rows"], axes=[3], keepdims=0),
        helper.make_node("Div", ["rows", "c255"], ["unit"]),
        helper.make_node("Transpose", ["unit"], ["output"], perm=[0, 2, 1]),
    ]
    constants = {"c255": np.float32(255)}
    return _save(path, nodes, constants, "input", "output", ["N", "T", 3])


def _save(
    path: Path,
    nodes: list[onnx.NodeProto],
    constants: dict[str, np.ndarray],
    input_name: str,
    output_name: str,
    out_shape: list[int | str],
) -> Path:
    graph = helper.make_graph(
        nodes,
        "stand-in",
        [
            helper.make_tensor_value_info(
                input_name, TensorProto.UINT8, ["N", 3, "T", "T"]
            )
        ],
        [helper.make_tensor_value_info(output_name, TensorProto.FLOAT, out_shape)],
        [numpy_helper.from_array(np.asarray(v), k) for k, v in constants.items()],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", OPSET)],
        ir_version=IR_VERSION,
    )
    onnx.checker.check_model(model)
    onnx.save(model, path)
    return path
