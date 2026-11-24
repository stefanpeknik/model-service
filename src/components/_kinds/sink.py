"""Sink: collects a run's batches into one artifact in a data format."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field

from components._kinds.component import Component
from components._kinds.options import Options
from components._kinds.slots import SINKS


if TYPE_CHECKING:
    from pathlib import Path

    from components._kinds.columns import Batch
    from components._kinds.run import Run
    from components._kinds.slots import Slot


class SinkOptions(Options):
    """Options every sink has.

    Attributes:
        output: The output name callers and providers refer to.
    """

    output: str = Field(description="The output name callers and providers refer to.")


class Sink(Component):
    """Writes a run's results into a file of one data format.

    Attributes:
        format: The data format the artifact has, e.g. `score_grid`.
        frame_reads: The source frame fields the sink needs before any item.
        ordered: True if rows must arrive in key order.
    """

    kind: ClassVar[str] = "sink"
    folder: ClassVar[str] = "sinks"
    slots: ClassVar[tuple[Slot, ...]] = (SINKS,)
    Options: ClassVar[type[Options]] = SinkOptions
    format: ClassVar[str]
    frame_reads: ClassVar[frozenset[str]] = frozenset()
    ordered: ClassVar[bool] = False
    options: SinkOptions

    @abstractmethod
    def open(self, frame: dict[str, Any], run: Run) -> None:
        """Start collecting; called once, before any batch.

        Args:
            frame: The source's frame row.
            run: The run's `Run`: `run.work_dir` is where to write, and
                `run.columns` gives the shapes of the columns it reads.
        """

    @abstractmethod
    def add(self, batch: Batch) -> None:
        """Collect one batch.

        Args:
            batch: Rows carrying at least the declared `reads`.
        """

    @abstractmethod
    def close(self) -> Path:
        """Finish the artifact; called once, even when no batch came.

        Returns:
            The path of the written artifact.
        """

    def abort(self) -> None:
        """Release what was collected; called instead of `close` when the run fails.

        The runner removes `run.work_dir` afterwards, so only resources held
        elsewhere (temporary files, memory maps) need releasing here.
        """
