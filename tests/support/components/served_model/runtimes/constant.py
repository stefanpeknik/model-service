from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import Field

from model_service.components import Options, Runtime


if TYPE_CHECKING:
    from pathlib import Path

    from model_service.components import BatchFunction, RuntimeContext


class ConstantOptions(Options):
    value: float = Field(0.5, ge=0, le=1, description="The score every tile gets.")
    threads: int = Field(1, ge=1, description="Threads the runtime may use.")


class Constant(Runtime):
    """Test runtime: scores every tile with the same value."""

    name = "constant"
    Options = ConstantOptions

    def load(self, path: Path, context: RuntimeContext) -> BatchFunction:
        """Ignore the weights and return a constant model."""
        assert isinstance(self.options, ConstantOptions)
        value = self.options.value
        return lambda tiles: np.full((len(tiles), 1), value, dtype=np.float32)
