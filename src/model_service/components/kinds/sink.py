"""Sink: collects a run's batches into one artifact in a data format."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field

from model_service.components.component import Component
from model_service.components.options import Options
from model_service.components.slots import SINKS


if TYPE_CHECKING:
    from pathlib import Path

    from model_service.components.columns import Batch
    from model_service.components.slots import Slot


class SinkOptions(Options):
    """Options every sink has.

    Attributes:
        output: The output name callers and exposure refer to.
    """

    output: str = Field(description="The output name callers and exposure refer to.")


class Sink(Component):
    """Writes a run's results into a file of one data format.

    Attributes:
        format: The data format the artifact has, e.g. `score_grid`.
        frame_reads: The source frame fields the sink needs before any item.
        ordered: True if rows must arrive in key order.
    """

    kind: ClassVar[str] = "sink"
    slots: ClassVar[tuple[Slot, ...]] = (SINKS,)
    Options: ClassVar[type[Options]] = SinkOptions
    format: ClassVar[str]
    frame_reads: ClassVar[frozenset[str]] = frozenset()
    ordered: ClassVar[bool] = False
    options: SinkOptions

    @abstractmethod
    def open(self, frame: dict[str, Any], run: Any) -> None:
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
