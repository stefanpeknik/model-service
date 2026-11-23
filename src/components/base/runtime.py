"""Runtime: turns local weights into a batch function."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from components.base.component import Component
from components.base.slots import RUNTIME


if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from components.base.slots import Slot


type BatchFunction = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class RuntimeContext:
    """What the served model tells its runtime when loading.

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


@dataclass(frozen=True)
class StandIn:
    """Tiny weights a runtime loads in the conformance suite, and a batch for them.

    Attributes:
        weights: The weights file or folder.
        batch: A batch the loaded function takes.
    """

    weights: Path
    batch: np.ndarray


class Runtime(Component):
    """Runs a model's weights (ONNX Runtime, timm, ...) inside a served model."""

    kind: ClassVar[str] = "runtime"
    slots: ClassVar[tuple[Slot, ...]] = (RUNTIME,)

    @abstractmethod
    def load(self, path: Path, context: RuntimeContext) -> BatchFunction:
        """Load the weights fetched by the served model's loader.

        Args:
            path: The local path the loader returned.
            context: The model's cache directory and batch geometry.

        Returns:
            A function from a batch of model inputs to a batch of outputs.
        """

    @classmethod
    def stand_in(cls, directory: Path) -> StandIn:
        """Write tiny weights this runtime loads, for the conformance suite.

        The default, an empty file and two black 8 px uint8 tiles, fits a
        runtime that ignores its weights; a runtime of a weights format
        writes a small model in it.

        Args:
            directory: An empty directory to write into.

        Returns:
            The weights and a batch for the function `load` returns.
        """
        import numpy as np

        weights = directory / "weights.bin"
        weights.write_bytes(b"")
        return StandIn(weights, np.zeros((2, 3, 8, 8), dtype=np.uint8))

    def active_providers(self) -> list[str]:
        """The execution providers actually in use after `load`.

        A runtime that can silently fall back (TensorRT to CUDA to CPU)
        reports what it got, so the fallback is visible in `describe()`.

        Returns:
            Provider names, best first; empty if the runtime has no such notion.
        """
        return []
