"""The `timm` runtime: a PyTorch model built by timm from a Hub-style folder."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import Field

from components.base import Runtime, StandIn
from components.base.options import Options


if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from components.base import BatchFunction, RuntimeContext


# Constructor options name classes; files name them by these keys.
MLP_LAYERS = {"swiglu_packed": ("timm.layers.mlp", "SwiGLUPacked")}
ACT_LAYERS = {"silu": ("torch.nn", "SiLU")}


class TimmOptions(Options):
    """Options of the `timm` runtime.

    Attributes:
        num_classes: Passed to the constructor, if given.
        mlp_layer: The MLP block class, by name, if given.
        act_layer: The activation class, by name, if given.
        autocast: Mixed precision for the forward pass.
        transform: Which per-tile transform turns tiles into model input.
    """

    num_classes: int | None = Field(
        None,
        ge=0,
        description=(
            "Classifier outputs, passed to timm's constructor; 0 removes the head, "
            "so the model returns its features. Left out, the folder's config decides."
        ),
    )
    mlp_layer: Literal["swiglu_packed"] | None = Field(
        None,
        description=(
            "The MLP block, passed to timm's constructor: `swiglu_packed` "
            "(timm's SwiGLUPacked, as Virchow2 needs). Left out, timm's default."
        ),
    )
    act_layer: Literal["silu"] | None = Field(
        None,
        description=(
            "The activation, passed to timm's constructor: `silu` (torch's SiLU). "
            "Left out, timm's default."
        ),
    )
    autocast: Literal["fp16", "none"] = Field(
        "none",
        description=(
            "`fp16` runs the forward pass under autocast: float16 on a GPU, "
            "bfloat16 on CPU, which has no float16 autocast. `none` runs float32."
        ),
    )
    transform: Literal["pretrained", "none"] = Field(
        "none",
        description=(
            "`pretrained` applies the transform timm derives from the folder's "
            "config (resize, crop, normalise) to each uint8 tile. `none` passes "
            "the batch as given, e.g. as `pre:` steps left it."
        ),
    )


class Timm(Runtime):
    """Runs a timm model on PyTorch, on the GPU when there is one.

    `runtime: {use: timm, num_classes: 0, mlp_layer: swiglu_packed, act_layer:
    silu, autocast: fp16, transform: pretrained}` reproduces Virchow2. The
    weights are a folder laid out like a Hugging Face repository
    (`config.json` and the weights file), as the `huggingface` loader fetches
    it; timm reads the architecture and its arguments from `config.json`.

    With `transform: pretrained` the runtime takes uint8 RGB tiles and
    transforms each one; with `transform: none` it takes the float batch that
    `pre:` steps produced. The output is the model's, as float32.
    """

    name = "timm"
    Options = TimmOptions
    options: TimmOptions

    def load(self, path: Path, context: RuntimeContext) -> BatchFunction:
        """Build the model from the folder and load its weights.

        Args:
            path: The Hub-style folder.
            context: Unused; timm needs no cache.

        Returns:
            A function from a batch of tiles to the model's output.
        """
        import timm
        import torch

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = timm.create_model(
            f"local-dir:{path}", pretrained=True, **self._constructor_options()
        )
        model = model.to(self._device).eval()
        prepare = self._transform(model)
        device = self._device
        # PyTorch's CPU autocast has no float16.
        dtype = torch.float16 if device.type == "cuda" else torch.bfloat16
        autocast = self.options.autocast == "fp16"

        def run(batch: np.ndarray) -> np.ndarray:
            """Run the model on a batch.

            Args:
                batch: Tiles as the transform expects them.

            Returns:
                The model's output, float32.
            """
            tensors = prepare(batch).to(device)
            with (
                torch.inference_mode(),
                torch.autocast(device.type, dtype=dtype, enabled=autocast),
            ):
                output = model(tensors)
            return output.float().cpu().numpy()

        return run

    @classmethod
    def stand_in(cls, directory: Path) -> StandIn:
        """Write timm's tiny `test_vit` as a Hub-style folder with random weights.

        Args:
            directory: Where to write the folder.

        Returns:
            The folder and two float32 160 px tiles, the input `transform:
            none` passes through.
        """
        import json

        import numpy as np
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
        return StandIn(folder, np.zeros((2, *size), dtype=np.float32))

    def active_providers(self) -> list[str]:
        """The device the model runs on, so a missing GPU shows.

        Returns:
            `["cuda"]` or `["cpu"]`.
        """
        return [self._device.type]

    def _constructor_options(self) -> dict[str, Any]:
        """The options for timm's constructor, names resolved to classes."""
        import importlib

        options: dict[str, Any] = {}
        if self.options.num_classes is not None:
            options["num_classes"] = self.options.num_classes
        for key, names, choice in (
            ("mlp_layer", MLP_LAYERS, self.options.mlp_layer),
            ("act_layer", ACT_LAYERS, self.options.act_layer),
        ):
            if choice is not None:
                module, attribute = names[choice]
                options[key] = getattr(importlib.import_module(module), attribute)
        return options

    def _transform(self, model: Any) -> Any:
        """The function turning a NumPy batch into the model's input tensor.

        Args:
            model: The timm model, whose pretrained config gives the transform.

        Returns:
            A function from a NumPy batch to a tensor.
        """
        import numpy as np
        import torch

        if self.options.transform == "none":
            return lambda batch: torch.from_numpy(np.ascontiguousarray(batch))

        from PIL import Image
        from timm.data.config import resolve_data_config
        from timm.data.transforms_factory import create_transform

        transform = create_transform(
            **resolve_data_config(model.pretrained_cfg, model=model)
        )

        def pretrained(batch: np.ndarray) -> Any:
            """Transform each uint8 tile as timm's pretrained config says.

            Args:
                batch: uint8 RGB tiles `(N, 3, H, W)`.

            Returns:
                The stacked transformed tiles.
            """
            images = (
                Image.fromarray(np.ascontiguousarray(tile.transpose(1, 2, 0)))
                for tile in batch
            )
            return torch.stack([transform(image) for image in images])

        return pretrained
