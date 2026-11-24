"""The `my_component` sink: <what file it writes from the rows>.

Copy to `src/components/sinks/<name>.py` and replace
`my_component`, `MyComponent` and the placeholders in <angle brackets>. A new
file layout also needs a module in `src/model_service/formats/`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from components.base import Col, Sink
from model_service.formats import value


if TYPE_CHECKING:
    from pathlib import Path

    from components.base import Batch, Run


class MyComponent(Sink):
    """<One sentence: what it writes, in which data format>.

    `sinks: [{use: my_component, output: <name>}]`. Writes `<output>.json` in
    the `value` format: <what the value means>.
    """

    name = "my_component"
    format = value.FORMAT.name
    example_options: ClassVar[dict[str, Any]] = {"output": "mean_score"}
    reads: ClassVar[dict[str, Col]] = {"scores": Col("float32", None)}

    def open(self, frame: dict[str, Any], run: Run) -> None:
        """Start collecting; called once, before any batch.

        Args:
            frame: The source's frame row.
            run: The run; write into `run.work_dir`.
        """
        self._path = run.work_dir / f"{self.options.output}.{value.FORMAT.extension}"
        self._total = 0.0
        self._count = 0

    def add(self, batch: Batch) -> None:
        """Collect one batch.

        Args:
            batch: Rows with the declared `reads`.
        """
        scores = batch["scores"].reshape(len(batch["scores"]), -1)
        self._total += float(np.sum(scores.mean(axis=1)))
        self._count += len(scores)

    def close(self) -> Path:
        """Write the file; called once, even when no batch came.

        Returns:
            The written file.
        """
        value.write(self._path, self._total / self._count if self._count else 0.0)
        return self._path
