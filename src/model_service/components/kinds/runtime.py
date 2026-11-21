"""Runtime: turns local weights into a batch function."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, ClassVar

from model_service.components.component import Component
from model_service.components.slots import RUNTIME


if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from model_service.components.slots import Slot


type BatchFunction = Callable[[np.ndarray], np.ndarray]


class Runtime(Component):
    """Runs a model's weights (ONNX Runtime, timm, ...) inside a served model."""

    kind: ClassVar[str] = "runtime"
    slots: ClassVar[tuple[Slot, ...]] = (RUNTIME,)

    @abstractmethod
    def load(self, path: Path) -> BatchFunction:
        """Load the weights fetched by the served model's loader.

        Args:
            path: The local path the loader returned.

        Returns:
            A function from a batch of model inputs to a batch of outputs.
        """
