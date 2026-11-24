"""The `my_component` runtime: <the library that runs the weights>.

Copy to `src/components/runtimes/<name>.py` and
replace `my_component`, `MyComponent` and the placeholders in <angle brackets>.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import Field

from components import Options, Runtime


if TYPE_CHECKING:
    from pathlib import Path

    from components import BatchFunction, RuntimeContext


class MyComponentOptions(Options):
    """Options of the `my_component` runtime.

    Attributes:
        threads: <A setting of the library>.
    """

    threads: int = Field(1, ge=1, description="<What the option changes>.", strict=True)


class MyComponent(Runtime):
    """<One sentence: what weights it runs and on what hardware>.

    `runtime: {use: my_component}`. <What the batch it takes looks like, e.g.
    uint8 RGB tiles `(N, 3, tile_size, tile_size)`, and what it returns.>
    """

    name = "my_component"
    Options = MyComponentOptions
    options: MyComponentOptions

    def load(self, path: Path, context: RuntimeContext) -> BatchFunction:
        """Load the weights the loader fetched.

        Args:
            path: The local weights; import the library inside this method.
            context: The model's cache directory and batch geometry.

        Returns:
            A function from a batch of tiles to one output per tile.
        """

        def run(batch: np.ndarray) -> np.ndarray:
            # Replace with the model's forward pass.
            means = batch.reshape(len(batch), -1).mean(axis=1) / 255
            return means.astype(np.float32).reshape(len(batch), 1, 1, 1)

        return run

    def active_providers(self) -> list[str]:
        """The devices or providers in use after `load`, so a fallback shows.

        Returns:
            Names, best first.
        """
        return ["cpu"]
