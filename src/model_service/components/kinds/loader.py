"""Loader: fetches weights to a local path."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar

from model_service.components.component import Component
from model_service.components.slots import STEP_WEIGHTS, WEIGHTS


if TYPE_CHECKING:
    from pathlib import Path

    from model_service.components.slots import Slot


class Loader(Component):
    """Fetches a model's weights (an MLflow artifact, a Hugging Face repo, ...).

    Used under a served model's `weights:` and under the `weights:` of an
    analysis step that carries its own small model.
    """

    kind: ClassVar[str] = "loader"
    slots: ClassVar[tuple[Slot, ...]] = (WEIGHTS, STEP_WEIGHTS)

    @abstractmethod
    def fetch(self, cache_dir: Path) -> Path:
        """Make the weights available locally.

        Args:
            cache_dir: A directory the loader may download into and reuse.

        Returns:
            The local path of the weights (a file or a directory).
        """
