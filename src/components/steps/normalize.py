"""The `normalize` step: uint8 tiles to float32, scaled to [0, 1] and standardised."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

import numpy as np
from pydantic import Field, model_validator

from components import Col, Options, Step


if TYPE_CHECKING:
    from components import Batch


PRESETS = {"imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))}

type Triple = tuple[float, float, float]


class NormalizeOptions(Options):
    """Options of the `normalize` step: a preset, or mean and std.

    Attributes:
        preset: Named mean and std.
        mean: Per-channel mean, on the [0, 1] scale.
        std: Per-channel standard deviation, on the [0, 1] scale.
    """

    preset: Literal["imagenet"] | None = Field(
        None,
        description=(
            "Named mean and std: `imagenet` is mean (0.485, 0.456, 0.406), "
            "std (0.229, 0.224, 0.225). Give either this or `mean` and `std`."
        ),
    )
    mean: Triple | None = Field(
        None, description="Per-channel (RGB) mean, on the [0, 1] scale."
    )
    std: Triple | None = Field(
        None,
        description="Per-channel (RGB) standard deviation, on the [0, 1] scale.",
    )

    @model_validator(mode="after")
    def _one_source(self) -> Self:
        """Require a preset or both mean and std, not both kinds."""
        explicit = (self.mean is not None, self.std is not None)
        if self.preset is not None and any(explicit):
            raise ValueError("give `preset` or `mean` and `std`, not both")
        if self.preset is None and not all(explicit):
            raise ValueError("give `preset`, or both `mean` and `std`")
        return self

    def values(self) -> tuple[Triple, Triple]:
        """The mean and std to apply.

        Returns:
            The preset's values, or the given ones.
        """
        if self.preset is not None:
            return PRESETS[self.preset]
        assert self.mean is not None and self.std is not None
        return self.mean, self.std


class Normalize(Step):
    """Turns uint8 RGB tiles into the float32 input most PyTorch models expect.

    `pre: [{use: normalize, preset: imagenet}]` divides by 255, subtracts the
    mean and divides by the std per channel, as torchvision's `ToTensor`
    followed by `Normalize` does, and with the same float32 arithmetic, so
    the result matches it bit for bit.
    """

    name = "normalize"
    Options = NormalizeOptions
    example_options: ClassVar[dict[str, Any]] = {"preset": "imagenet"}
    reads: ClassVar[dict[str, Col]] = {"tile": Col("uint8", (3, None, None))}
    adds: ClassVar[dict[str, Col]] = {"tile": Col("float32", (3, None, None))}
    options: NormalizeOptions

    def __call__(self, batch: Batch) -> Batch:
        """Scale and standardise every tile.

        Args:
            batch: `tile`, uint8 RGB `(rows, 3, height, width)`.

        Returns:
            The batch with `tile` as float32 of the same shape.
        """
        mean, std = (
            np.asarray(v, np.float32)[:, None, None] for v in self.options.values()
        )
        scaled = batch["tile"].astype(np.float32) / np.float32(255)
        return batch | {"tile": (scaled - mean) / std}
