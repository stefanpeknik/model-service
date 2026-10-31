from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, TypedDict

import lz4.frame
import numpy as np
from fastapi import FastAPI, Request, Response
from ray import serve


if TYPE_CHECKING:
    import torch
    from numpy.typing import NDArray


SUMMARIES = ("tile",)


class Config(TypedDict):
    tile_size: int
    model: dict[str, Any]
    max_batch_size: int
    batch_wait_timeout_s: float


fastapi = FastAPI()


@serve.deployment(num_replicas="auto")
@serve.ingress(fastapi)
class ProvGigaPath:
    """GigaPath tile encoder for pathology."""

    model: torch.nn.Module
    transforms: Any
    tile_size: int

    def __init__(self) -> None:
        import torch

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def reconfigure(self, config: Config) -> None:
        import timm
        from torchvision import transforms

        self.tile_size = config["tile_size"]
        model_config = dict(config["model"])
        repo_id = model_config["repo_id"]

        self.model = timm.create_model(
            f"hf_hub:{repo_id}",
            pretrained=True,
        )
        self.model = self.model.to(self.device).eval()

        # Based on the HF documentation
        self.transforms = transforms.Compose(
            [
                transforms.Resize(
                    256, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

        self.predict.set_max_batch_size(config["max_batch_size"])  # type: ignore[attr-defined]
        self.predict.set_batch_wait_timeout_s(config["batch_wait_timeout_s"])  # type: ignore[attr-defined]

    @serve.batch
    async def predict(self, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
        import torch

        tensors = torch.stack(inputs).to(self.device)
        with torch.inference_mode():
            output = self.model(tensors)
        return list(output)

    def get_config(self) -> dict[str, Any]:
        return {"tile_size": self.tile_size, "summaries": list(SUMMARIES)}

    async def embed(self, tile: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Embed one CHW uint8 tile as a `(dim,)` float32 vector."""
        output = await self._encode(np.ascontiguousarray(tile.transpose(1, 2, 0)))
        return output.float().cpu().numpy()

    async def _encode(self, image: NDArray[np.uint8]) -> torch.Tensor:
        from PIL import Image

        return await self.predict(self.transforms(Image.fromarray(image)))

    @fastapi.post("/")
    async def root(self, request: Request) -> Response:
        data = await asyncio.to_thread(lz4.frame.decompress, await request.body())
        image = np.frombuffer(data, dtype=np.uint8).reshape(
            self.tile_size, self.tile_size, 3
        )

        output_dtype = np.dtype(
            request.headers.get("x-output-dtype", "float32").lower()
        )

        raw_output = await self._encode(image)
        result = raw_output.cpu().numpy().astype(output_dtype, copy=False)
        output_shape = str(result.shape)

        return Response(
            content=lz4.frame.compress(result.tobytes()),
            media_type="application/octet-stream",
            headers={
                "x-output-shape": output_shape,
            },
        )


app = ProvGigaPath.bind()  # type: ignore[attr-defined]
