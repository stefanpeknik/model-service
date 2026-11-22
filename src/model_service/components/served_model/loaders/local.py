"""The `local` loader: weights already on this machine."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field

from model_service.components.kinds import Loader
from model_service.components.options import Options


class LocalOptions(Options):
    """Options of the `local` loader.

    Attributes:
        path: The weights file or directory.
    """

    path: str = Field(
        min_length=1,
        description="The weights file or directory on the machine running the model.",
    )


class Local(Loader):
    """Weights at a path on the machine running the model, for tests and trials.

    `weights: {use: local, path: /data/model.onnx}`. Nothing is downloaded; on
    the cluster the path must be on a mounted volume.
    """

    name = "local"
    Options = LocalOptions
    example_options: ClassVar[dict[str, Any]] = {"path": "/data/model.onnx"}
    options: LocalOptions

    def fetch(self, cache_dir: Path) -> Path:
        """Return the configured path.

        Args:
            cache_dir: Unused; the weights are already local.

        Returns:
            The configured path.
        """
        return Path(self.options.path)

    def revision(self) -> str:
        """Name the weights by their path.

        Returns:
            The configured path.
        """
        return self.options.path
