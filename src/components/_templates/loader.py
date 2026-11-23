"""The `my_component` loader: <where the weights come from>.

Copy to `src/components/loaders/<name>.py` and
replace `my_component`, `MyComponent` and the placeholders in <angle brackets>.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field

from components.base import Loader, Options


class MyComponentOptions(Options):
    """Options of the `my_component` loader.

    Attributes:
        path: <What names the weights: a URI, a repository, a path>.
    """

    path: str = Field(min_length=1, description="<What names the weights>.")


class MyComponent(Loader):
    """<One sentence: where it fetches weights from>.

    `weights: {use: my_component, path: <example>}`. <What the deployment
    environment must provide, e.g. a token or a tracking URI.>
    """

    name = "my_component"
    Options = MyComponentOptions
    example_options: ClassVar[dict[str, Any]] = {"path": "/data/model.onnx"}
    options: MyComponentOptions

    def fetch(self, cache_dir: Path) -> Path:
        """Make the weights available locally.

        Args:
            cache_dir: A directory to download into; import download
                libraries inside this method, not at the top of the module.

        Returns:
            The local path of the weights.
        """
        return Path(self.options.path)

    def revision(self) -> str:
        """Name the exact weights, for `describe()`.

        Returns:
            A pinned reference, e.g. a URI with a commit or checksum.
        """
        return self.options.path
