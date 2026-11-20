import asyncio
import os
from typing import Any, NotRequired, TypedDict

import numpy as np
from fastapi import FastAPI, Request
from numpy.typing import NDArray
from ray import serve


class Config(TypedDict):
    tile_size: int
    model: dict[str, Any]
    max_batch_size: int
    batch_wait_timeout_s: float
    trt_cache_path: str
    intra_op_num_threads: int

    trt_max_workspace_size: NotRequired[int]
    trt_builder_optimization_level: NotRequired[int]


fastapi = FastAPI()


@serve.deployment(num_replicas="auto")
@serve.ingress(fastapi)
class BinaryClassifier:
    """Binary classifier for tissue tiles using ONNX Runtime with GPU support."""

    tile_size: int

    def __init__(self) -> None:
        import lz4.frame

        self.lz4 = lz4.frame

    def reconfigure(self, config: Config) -> None:
        """Load the ONNX model and configure inference settings."""
        import importlib

        import onnxruntime as ort

        self.tile_size = config["tile_size"]

        cache_path = config["trt_cache_path"]
        os.makedirs(cache_path, exist_ok=True)

        min_shape = f"input:1x3x{self.tile_size}x{self.tile_size}"
        opt_shape = (
            f"input:{config['max_batch_size']}x3x{self.tile_size}x{self.tile_size}"
        )
        max_shape = (
            f"input:{config['max_batch_size']}x3x{self.tile_size}x{self.tile_size}"
        )

        # TensorRT optimization options:
        # - trt_fp16_enable: Enable FP16 mode for faster inference on Tensor Cores (default: False is slower)
        # - trt_engine_cache_enable: Cache TensorRT engines to disk to avoid rebuilding on restart (default: False rebuilds every time)
        # - trt_engine_cache_path: Directory to store cached engines
        # - trt_timing_cache_enable: Cache kernel timing info to speed up subsequent engine builds (default: False is slower)
        # - trt_builder_optimization_level: TensorRT builder optimization level taken from config (defaults to 1, which is faster to build but less optimized; levels are 1-5)
        # - trt_max_workspace_size: Memory available for TensorRT to find optimal kernels (default: 1GB)
        #   Default 1GB is insufficient for high-resolution processing, restricting valid kernels.
        #   We default to 8GB as a reasonable balance, but can be overridden via config.
        trt_options = {
            "device_id": 0,
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": cache_path,
            "trt_max_workspace_size": config.get("trt_max_workspace_size", 8 * 1024**3),
            "trt_builder_optimization_level": config.get(
                "trt_builder_optimization_level", 1
            ),
            "trt_timing_cache_enable": True,
            "trt_profile_min_shapes": min_shape,
            "trt_profile_max_shapes": max_shape,
            "trt_profile_opt_shapes": opt_shape,
        }

        # Configure ONNX Runtime session
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = config["intra_op_num_threads"]
        sess_options.inter_op_num_threads = 1

        # Enable all graph optimizations (constant folding, node fusion, etc.) for maximum inference performance.
        # ORT_SEQUENTIAL ensures ops run one at a time within a session, which avoids inter-op parallelism
        # overhead and is preferred when batching is handled externally (as done here via @serve.batch).
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        # Load model from provider (e.g., MLflow)
        model_config = dict(config["model"])
        module_path, attr_name = model_config.pop("_target_").split(":")
        provider = getattr(importlib.import_module(module_path), attr_name)

        self.session = ort.InferenceSession(
            provider(**model_config),
            providers=[
                (
                    "TensorrtExecutionProvider",
                    trt_options,
                ),
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
            session_options=sess_options,
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.predict.set_max_batch_size(config["max_batch_size"])  # type: ignore[attr-defined]
        self.predict.set_batch_wait_timeout_s(config["batch_wait_timeout_s"])  # type: ignore[attr-defined]

    @serve.batch
    async def predict(self, images: list[NDArray[np.uint8]]) -> list[float]:
        batch = np.stack(images, axis=0, dtype=np.uint8)

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: batch},
        )

        return outputs[0].flatten().tolist()  # pyright: ignore[reportAttributeAccessIssue]

    @fastapi.post("/")
    async def root(self, request: Request) -> float:
        data = await asyncio.to_thread(self.lz4.decompress, await request.body())

        image = (
            np.frombuffer(data, dtype=np.uint8)
            .reshape(self.tile_size, self.tile_size, 3)
            .transpose(2, 0, 1)
        )

        result = await self.predict(image)
        return result


app = BinaryClassifier.bind()  # type: ignore[attr-defined]
