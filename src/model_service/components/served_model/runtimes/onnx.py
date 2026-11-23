"""The `onnx` runtime: an ONNX graph on ONNX Runtime, with TensorRT on or off."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field, ValidationInfo, field_validator

from model_service.components.kinds import Runtime, StandIn
from model_service.components.options import Options


if TYPE_CHECKING:
    from pathlib import Path

    from model_service.components.kinds import BatchFunction, RuntimeContext

logger = logging.getLogger(__name__)
TENSORRT = "TensorrtExecutionProvider"


class OnnxOptions(Options):
    """Options of the `onnx` runtime.

    Attributes:
        tensorrt: Whether to try TensorRT first.
        workspace_gib: TensorRT's workspace for choosing kernels.
        optimization_level: TensorRT's builder optimisation level.
        fp16: Whether TensorRT may use FP16 kernels.
        apply_session_options: Whether the session gets `threads`, all graph
            optimisations and sequential execution, or ONNX Runtime's defaults.
        threads: ONNX Runtime's intra-op threads.
        output_dtype: The dtype the graph's output is cast to, if any.
    """

    tensorrt: bool = Field(
        False,
        description=(
            "Try TensorRT first, then CUDA, then CPU. The engine is built on the "
            "first start and cached; `describe()` shows which provider is active."
        ),
    )
    workspace_gib: float = Field(
        8,
        gt=0,
        description=(
            "GPU memory TensorRT may use to choose kernels; too little rules out "
            "the fast ones for large tiles."
        ),
    )
    optimization_level: int = Field(
        1,
        ge=0,
        le=5,
        description=(
            "TensorRT builder optimisation level: higher builds slower and may "
            "run faster."
        ),
    )
    fp16: bool = Field(True, description="Let TensorRT use FP16 kernels.")
    apply_session_options: bool = Field(
        True,
        description=(
            "Give the session `threads`, all graph optimisations and sequential "
            "execution. `false` leaves ONNX Runtime's defaults (intra-op threads "
            "for every core)."
        ),
    )
    threads: int = Field(1, ge=1, description="ONNX Runtime's intra-op threads.")
    output_dtype: Literal["float32", "float16"] | None = Field(
        None,
        description=(
            "Cast the graph's output to this dtype, e.g. `float16` to halve a "
            "score map; left out, the output keeps the graph's dtype."
        ),
    )

    @field_validator("threads")
    @classmethod
    def _threads_need_session_options(cls, threads: int, info: ValidationInfo) -> int:
        if not info.data.get("apply_session_options", True):
            raise ValueError(
                "threads has no effect with apply_session_options: false; "
                "remove threads or apply the session options"
            )
        return threads


class Onnx(Runtime):
    """Runs an ONNX graph on ONNX Runtime, optionally through TensorRT.

    The graph takes the batch as the host gives it (uint8 RGB tiles,
    `(N, 3, tile_size, tile_size)`, unless `pre:` changes them) as its first
    input and returns its first output. Input and output names are read from
    the graph. The TensorRT engine cache lives in the served model's cache
    directory, so a changed model never reuses a stale engine.
    """

    name = "onnx"
    Options = OnnxOptions
    options: OnnxOptions

    def load(self, path: Path, context: RuntimeContext) -> BatchFunction:
        """Open an inference session on the graph.

        Args:
            path: The ONNX file.
            context: The cache directory and the batch geometry for TensorRT's
                shape profile.

        Returns:
            A function running the graph's first output on a batch.
        """
        import onnxruntime as ort

        providers: list[Any] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if self.options.tensorrt:
            providers.insert(0, (TENSORRT, self._tensorrt(path, context)))
        session = ort.InferenceSession(
            str(path), providers=providers, sess_options=self._session_options()
        )
        self._session = session
        if self.options.tensorrt and TENSORRT not in session.get_providers():
            logger.warning(
                "tensorrt: true, but ONNX Runtime runs %s on %s without %s",
                path.name,
                session.get_providers(),
                TENSORRT,
            )
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        dtype = self.options.output_dtype

        def run(batch: Any) -> Any:
            output = session.run([output_name], {input_name: batch})[0]
            return output if dtype is None else output.astype(dtype)

        return run

    @classmethod
    def stand_in(cls, directory: Path) -> StandIn:
        """Write an ONNX graph that scores each tile by its mean brightness.

        It needs the `onnx` package, which the dev dependency group has.

        Args:
            directory: Where to write `stand_in.onnx`.

        Returns:
            The graph and two uint8 8 px tiles.
        """
        import numpy as np
        import onnx
        from onnx import TensorProto, helper

        tiles = helper.make_tensor_value_info(
            "tiles", TensorProto.UINT8, ["n", 3, "t", "t"]
        )
        scores = helper.make_tensor_value_info(
            "scores", TensorProto.FLOAT, ["n", 1, 1, 1]
        )
        graph = helper.make_graph(
            [
                helper.make_node("Cast", ["tiles"], ["pixels"], to=TensorProto.FLOAT),
                helper.make_node(
                    "ReduceMean", ["pixels"], ["scores"], axes=[1, 2, 3], keepdims=1
                ),
            ],
            "stand_in",
            [tiles],
            [scores],
        )
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model.ir_version = 8
        path = directory / "stand_in.onnx"
        onnx.save(model, str(path))
        return StandIn(path, np.zeros((2, 3, 8, 8), dtype=np.uint8))

    def active_providers(self) -> list[str]:
        """The providers ONNX Runtime actually uses, e.g. after a fallback.

        Returns:
            Provider names, best first.
        """
        return list(self._session.get_providers())

    def _session_options(self) -> Any:
        """The session options, or None for ONNX Runtime's defaults."""
        if not self.options.apply_session_options:
            return None
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self.options.threads
        sess_options.inter_op_num_threads = 1
        # Batching happens outside the session, so ops run one at a time.
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        return sess_options

    def _tensorrt(self, path: Path, context: RuntimeContext) -> dict[str, Any]:
        """TensorRT provider options with a shape profile for the graph's input."""
        import onnxruntime as ort

        # The profile names the input, which only a session can read; a bare CPU
        # session without optimisations reads it cheaply.
        probe_options = ort.SessionOptions()
        probe_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
        probe = ort.InferenceSession(
            str(path), providers=["CPUExecutionProvider"], sess_options=probe_options
        )
        input_name = probe.get_inputs()[0].name
        del probe

        context.cache_dir.mkdir(parents=True, exist_ok=True)
        tile = f"3x{context.tile_size}x{context.tile_size}"
        largest = f"{input_name}:{context.max_batch_size}x{tile}"
        return {
            "device_id": 0,
            "trt_fp16_enable": self.options.fp16,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(context.cache_dir),
            "trt_max_workspace_size": int(self.options.workspace_gib * 1024**3),
            "trt_builder_optimization_level": self.options.optimization_level,
            "trt_timing_cache_enable": True,
            "trt_profile_min_shapes": f"{input_name}:1x{tile}",
            "trt_profile_max_shapes": largest,
            "trt_profile_opt_shapes": largest,
        }
