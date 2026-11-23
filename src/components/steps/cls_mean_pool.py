"""The `cls_mean_pool` step: a ViT's class token and the mean of its patch tokens."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from pydantic import Field

from components.base import Step
from components.base.columns import Col
from components.base.options import Options


if TYPE_CHECKING:
    from components.base.columns import Batch


class ClsMeanPoolOptions(Options):
    """Options of the `cls_mean_pool` step.

    Attributes:
        registers: Register tokens between the class token and the patch tokens.
    """

    registers: int = Field(
        0,
        ge=0,
        description=(
            "Register tokens after the class token, left out of the mean and of "
            "unpooled tokens (Virchow2 has 4)."
        ),
    )


class ClsMeanPool(Step):
    """Pools a ViT's tokens into two vectors: the class token and the patch mean.

    `post: [{use: cls_mean_pool, registers: 4}]` turns Virchow2's tokens
    `(1 + registers + patches, dim)` into `(2, dim)`, the summaries
    `[global, local_mean]`. It runs in the served model, so only the two
    vectors leave the replica.

    The mean is computed with PyTorch, the way the encoder's own code pools,
    so the features match it bit for bit; PyTorch must be installed where
    the step runs. A caller that skips pooling gets the class token and the
    patch tokens, without the registers.
    """

    name = "cls_mean_pool"
    Options = ClsMeanPoolOptions
    example_options: ClassVar[dict[str, Any]] = {"registers": 4}
    reads: ClassVar[dict[str, Col]] = {"output": Col("float32", (None, None))}
    adds: ClassVar[dict[str, Col]] = {"output": Col("float32", (2, None))}
    options: ClsMeanPoolOptions

    def __call__(self, batch: Batch) -> Batch:
        """Replace each row's tokens with its class token and patch mean.

        Args:
            batch: `output`, tokens `(rows, tokens, dim)`.

        Returns:
            The batch with `output` as `(rows, 2, dim)`.
        """
        import torch

        tokens = torch.from_numpy(batch["output"])
        patches = tokens[:, 1 + self.options.registers :]
        pooled = torch.stack([tokens[:, 0], patches.mean(dim=1)], dim=1)
        return batch | {"output": pooled.numpy()}

    def skipped(self, batch: Batch) -> Batch:
        """Drop the register tokens, which unpooled callers never got.

        Args:
            batch: `output`, tokens `(rows, tokens, dim)`.

        Returns:
            The batch with the class token and patch tokens in `output`.
        """
        registers = np.arange(1, 1 + self.options.registers)
        return batch | {"output": np.delete(batch["output"], registers, axis=1)}
