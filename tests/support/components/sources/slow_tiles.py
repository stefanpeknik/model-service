from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from pydantic import Field

from components.base import Col, Opened, Options, Source


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    from components.base import Batch


class SlowTilesOptions(Options):
    load_s: float = Field(0.5, ge=0, description="Seconds each load takes.")


class SlowTiles(Source):
    """Test source: blank tiles that load slowly, like a slide read.

    Closing it while a load is still running raises, as closing an OpenSlide
    handle under a running read crashes the process.
    """

    name = "slow_tiles"
    Options = SlowTilesOptions
    adds: ClassVar[dict[str, Col]] = {"tile": Col("uint8", (3, 64, 64))}
    frame: ClassVar[dict[str, Col]] = {
        "extent_x": Col("int64"),
        "extent_y": Col("int64"),
    }

    def inputs(self) -> dict[str, bool]:
        """A slide, which it never reads."""
        return {"slide": True}

    @contextmanager
    def open(self, inputs: dict[str, Path], run: Any) -> Iterator[Opened]:
        """Give eight keys whose loads sleep; refuse to close under a load."""
        assert isinstance(self.options, SlowTilesOptions)
        delay = self.options.load_s
        lock = threading.Lock()
        loading = 0

        def load(chosen: Sequence[int]) -> Batch:
            nonlocal loading
            with lock:
                loading += 1
            time.sleep(delay)
            with lock:
                loading -= 1
            return {"tile": np.zeros((len(chosen), 3, 64, 64), np.uint8)}

        frame = {"extent_x": 512, "extent_y": 64}
        try:
            yield Opened(frame=frame, keys=list(range(8)), load=load)
        finally:
            with lock:
                if loading:
                    raise RuntimeError(f"closed with {loading} loads running")
