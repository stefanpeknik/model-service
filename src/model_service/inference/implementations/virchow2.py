from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, TypedDict

import lz4.frame
import numpy as np
from fastapi import FastAPI, Request, Response
from ray import serve


if TYPE_CHECKING:
    import torch


class Config(TypedDict):
    tile_size: int
    model: dict[str, Any]
    max_batch_size: int
    batch_wait_timeout_s: float


fastapi = FastAPI()


@serve.deployment(num_replicas="auto")
@serve.ingress(fastapi)
class Virchow2:
    """Virchow2 foundation model for pathology."""

    model: torch.nn.Module
    transforms: Any
    tile_size: int

    def __init__(self) -> None:
        import torch

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reconfigure(self, config: Config) -> None:
        import timm
        import torch
        from timm.data.config import resolve_data_config
        from timm.data.transforms_factory import create_transform
        from timm.layers.mlp import SwiGLUPacked

        self.tile_size = config["tile_size"]
        model_config = dict(config["model"])
        repo_id = model_config["repo_id"]

        self.model = timm.create_model(
            f"hf-hub:{repo_id}",
            pretrained=True,
            num_classes=0,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        self.model = self.model.to(self.device).eval()

        self.transforms = create_transform(
            **resolve_data_config(self.model.pretrained_cfg, model=self.model)
        )

        self.predict.set_max_batch_size(config["max_batch_size"])  # type: ignore[attr-defined]
        self.predict.set_batch_wait_timeout_s(config["batch_wait_timeout_s"])  # type: ignore[attr-defined]

    @serve.batch
    async def predict(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        import torch

        tensors = torch.stack(inputs).to(self.device)
        device_type = self.device.type

        # PyTorch autocast does not support float16 on CPU (throws RuntimeError).
        # bfloat16 is the only supported low-precision option for CPU inference.
        autocast_dtype = torch.float16 if device_type == "cuda" else torch.bfloat16

        with (
            torch.inference_mode(),
            torch.autocast(device_type=device_type, dtype=autocast_dtype),
        ):
            output = self.model(tensors)

        return list(output)

    @fastapi.post("/")
    async def root(self, request: Request) -> Response:
        import torch
        from PIL import Image

        data = await asyncio.to_thread(lz4.frame.decompress, await request.body())
        image = np.frombuffer(data, dtype=np.uint8).reshape(
            self.tile_size, self.tile_size, 3
        )

        output_dtype = np.dtype(
            request.headers.get("x-output-dtype", "float32").lower()
        )
        pool_tokens = request.headers.get("x-pool-tokens", "true").lower() == "true"

        tensor = self.transforms(Image.fromarray(image))

        raw_output: torch.Tensor = await self.predict(tensor)

        if pool_tokens:
            class_token = raw_output[0]
            patch_tokens = raw_output[5:]
            result_tensor = torch.stack([class_token, patch_tokens.mean(dim=0)])
        else:
            result_tensor = torch.cat([raw_output[0:1], raw_output[5:]], dim=0)

        result = result_tensor.cpu().numpy().astype(output_dtype, copy=False)
        output_shape = str(result.shape)

        return Response(
            content=lz4.frame.compress(result.tobytes()),
            media_type="application/octet-stream",
            headers={
                "x-output-shape": output_shape,
            },
        )


app = Virchow2.bind()  # type: ignore[attr-defined]
