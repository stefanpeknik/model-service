"""The `sigmoid` step: turns each logit into a probability."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from components.base import Col, Options, Step


if TYPE_CHECKING:
    from components.base import Batch


class SigmoidOptions(Options):
    """Options of the `sigmoid` step: none."""


class Sigmoid(Step):
    """Turns logits into independent probabilities, `1 / (1 + exp(-x))` per value.

    `post: [{use: sigmoid}]` fixes a model that returns logits for one class,
    or for several classes that may all be present, so its file can declare
    `output: {range: [0, 1]}`. For classes that exclude each other, use
    `softmax`.
    """

    name = "sigmoid"
    Options = SigmoidOptions
    example_options: ClassVar[dict[str, Any]] = {}
    reads: ClassVar[dict[str, Col]] = {"output": Col("float32", None)}
    adds: ClassVar[dict[str, Col]] = {"output": Col("float32", None)}
    options: SigmoidOptions

    def __call__(self, batch: Batch) -> Batch:
        """Apply the logistic function to every value.

        Args:
            batch: `output`, logits of any shape.

        Returns:
            The batch with probabilities in `output`.
        """
        logits = batch["output"]
        # exp of a non-positive number only, so large logits cannot overflow.
        small = np.exp(-np.abs(logits))
        probabilities = np.where(logits >= 0, 1 / (1 + small), small / (1 + small))
        return batch | {"output": probabilities.astype(logits.dtype)}
