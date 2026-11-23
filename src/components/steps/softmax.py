"""The `softmax` step: turns a model's logits into probabilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from pydantic import Field

from components.base import Col, Options, Step


if TYPE_CHECKING:
    from components.base import Batch


class SoftmaxOptions(Options):
    """Options of the `softmax` step.

    Attributes:
        temperature: Divides the logits first.
    """

    temperature: float = Field(
        1.0, gt=0, description="Divides the logits first; above 1 flattens the scores."
    )


class Softmax(Step):
    """Turns per-channel logits into probabilities that sum to 1.

    `post: [{use: softmax}]` fixes a model that returns logits for classes
    that exclude each other, so its file can declare `output: {range: [0, 1]}`.
    The channels are the first axis of a score map `(channels, size, size)`;
    each pixel gets its own distribution.
    """

    name = "softmax"
    Options = SoftmaxOptions
    example_options: ClassVar[dict[str, Any]] = {"temperature": 1.0}
    reads: ClassVar[dict[str, Col]] = {"output": Col("float32", (None, None, None))}
    adds: ClassVar[dict[str, Col]] = {"output": Col("float32", (None, None, None))}
    options: SoftmaxOptions

    def __call__(self, batch: Batch) -> Batch:
        """Apply softmax over the channel axis.

        Args:
            batch: `output`, logits `(rows, channels, size, size)`.

        Returns:
            The batch with probabilities in `output`.
        """
        logits = batch["output"] / np.float32(self.options.temperature)
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return batch | {"output": exp / exp.sum(axis=1, keepdims=True)}
