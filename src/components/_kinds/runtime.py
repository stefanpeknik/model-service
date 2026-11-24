"""Runtime: turns local weights into a batch function."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from components._kinds.component import Component
from components._kinds.slots import RUNTIME


if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from components._kinds.slots import Slot


type BatchFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class RuntimeContext:
    """What the model tells its runtime when loading.

    Attributes:
        cache_dir: A directory for compiled artifacts (e.g. TensorRT engines),
            derived from the model ID and declaration digest, so a changed
            model never reuses a stale one. It may not exist yet.
        tile_size: The tile edge the model takes.
        max_batch_size: The largest batch the runtime will be given.
    """

    cache_dir: Path
    tile_size: int
    max_batch_size: int


class Runtime(Component):
    """Runs a model's weights (ONNX Runtime, timm, ...) inside the model host."""

    kind: ClassVar[str] = "runtime"
    folder: ClassVar[str] = "runtimes"
    slots: ClassVar[tuple[Slot, ...]] = (RUNTIME,)

    @abstractmethod
    def load(self, path: Path, context: RuntimeContext) -> BatchFunction:
        """Load the weights fetched by the model's loader.

        Args:
            path: The local path the loader returned.
            context: The model's cache directory and batch geometry.

        Returns:
            A function from a batch of model inputs to a batch of outputs.
        """

    def active_providers(self) -> list[str]:
        """The execution providers actually in use after `load`.

        A runtime that can silently fall back (TensorRT to CUDA to CPU)
        reports what it got, so the fallback is visible in `describe()`.

        Returns:
            Provider names, best first; empty if the runtime has no such notion.
        """
        return []
