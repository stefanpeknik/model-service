from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field

from components import Loader, Options


class LocalFileOptions(Options):
    path: str = Field(description="A weights file on this machine.")


class LocalFile(Loader):
    """Test loader: weights already on disk."""

    name = "local_file"
    Options = LocalFileOptions
    example_options: ClassVar[dict[str, Any]] = {"path": "weights.bin"}

    def fetch(self, cache_dir: Path) -> Path:
        """Return the configured path unchanged."""
        assert isinstance(self.options, LocalFileOptions)
        return Path(self.options.path)

    def revision(self) -> str:
        """Name the weights by their path."""
        assert isinstance(self.options, LocalFileOptions)
        return self.options.path
