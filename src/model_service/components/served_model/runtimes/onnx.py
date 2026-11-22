"""The `onnx` runtime: an ONNX graph on ONNX Runtime, with TensorRT on or off."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import Field

from model_service.components.kinds import Runtime
from model_service.components.options import Options


if TYPE_CHECKING:
    from pathlib import Path

    from model_service.components.kinds import BatchFunction, RuntimeContext


class OnnxOptions(Options):
    """Options of the `onnx` runtime.

    Attributes:
        tensorrt: Whether to try TensorRT first.
        workspace_gib: TensorRT's workspace for choosing kernels.
        optimization_level: TensorRT's builder optimisation level.
        fp16: Whether TensorRT may use FP16 kernels.
        threads: ONNX Runtime's intra-op threads.
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
    threads: int = Field(1, ge=1, description="ONNX Runtime's intra-op threads.")


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

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self.options.threads
        sess_options.inter_op_num_threads = 1
        # Batching happens outside the session, so ops run one at a time.
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        providers: list[Any] = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if self.options.tensorrt:
            providers.insert(
                0, ("TensorrtExecutionProvider", self._tensorrt(path, context))
            )
        session = ort.InferenceSession(
            str(path), providers=providers, sess_options=sess_options
        )
        self._session = session
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        def run(batch: Any) -> Any:
            return session.run([output_name], {input_name: batch})[0]

        return run

    def active_providers(self) -> list[str]:
        """The providers ONNX Runtime actually uses, e.g. after a fallback.

        Returns:
            Provider names, best first.
        """
        return list(self._session.get_providers())

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
