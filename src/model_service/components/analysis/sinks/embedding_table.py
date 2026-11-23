"""`embedding_table`: one feature row per tile, in grid order, as an HDF5 table.

The table's header comes from the source's frame (level, sizes, stride) and
the run's served model (its ID, summaries and mpp). Rows are written by one
thread of the sink's own, in the order they arrive, so the runner's event
loop never waits on HDF5.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, ClassVar

from model_service.components import Col, Sink
from model_service.contracts.failures import FailureCode, RunFailure
from model_service.formats import embedding_table


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from model_service.components import Batch


class EmbeddingTableSink(Sink):
    """Writes each tile's `features` and position as a row of an embedding table.

    Writes `<output>.h5` in the `embedding_table` format: tile corners on the
    level read and float32 features, one vector per summary of the served
    model, rows in grid order. Tiles the source skipped get no row; when it
    skipped all of them, the table is empty.
    """

    name = "embedding_table"
    format = embedding_table.NAME
    ordered = True
    example_options: ClassVar[dict[str, Any]] = {"output": "embeddings"}
    reads: ClassVar[dict[str, Col]] = {
        "tile_x": Col("int64"),
        "tile_y": Col("int64"),
        "features": Col("float32", None),
    }
    frame_reads = frozenset(
        {
            "extent_x",
            "extent_y",
            "tile_extent_x",
            "stride_x",
            "level",
            "mpp_x",
            "mpp_y",
            "downsample",
        }
    )

    def open(self, frame: dict[str, Any], run: Any) -> None:
        """Start the writer thread and create the table with its header.

        Args:
            frame: The source's frame row.
            run: The run's `Run`; `run.model` is the served model whose
                features the rows hold.

        Raises:
            ValueError: If the run has no served model.
        """
        model = run.model
        if model is None:
            raise ValueError("embedding_table needs a model step for its header")
        header = embedding_table.Header(
            model_id=model.id,
            summary_names=tuple(model.output.summaries),
            slide_level=frame["level"],
            level_downsample=frame["downsample"],
            slide_width=frame["extent_x"],
            slide_height=frame["extent_y"],
            mpp_x=frame["mpp_x"],
            mpp_y=frame["mpp_y"],
            model_mpp=model.input.mpp,
            tile_size=frame["tile_extent_x"],
            stride=frame["stride_x"],
        )
        self._path: Path = (
            run.work_dir / f"{self.options.output}.{embedding_table.EXTENSION}"
        )
        self._thread = ThreadPoolExecutor(max_workers=1)
        self._pending: list[Future[None]] = []
        self._writer: embedding_table.TableWriter | None = None

        def create() -> None:
            """Create the file on the writer thread, like every write after it."""
            self._writer = embedding_table.TableWriter(self._path, header)

        self._submit(create)

    def add(self, batch: Batch) -> None:
        """Queue a batch of rows for the writer thread.

        Args:
            batch: `tile_x`, `tile_y` and `features`, in key order.

        Raises:
            RunFailure: `invalid_model_output` if an earlier batch's
                features changed shape; any other failure of an earlier
                write as it was raised.
        """
        import numpy as np

        self._raise_failed()
        coordinates = np.stack([batch["tile_x"], batch["tile_y"]], axis=1)
        features = batch["features"]
        self._submit(lambda: self._table().append(coordinates, features))

    def close(self) -> Path:
        """Wait for every queued row and finish the table.

        Rows are queued as tiles finish, so little is left to wait for.

        Returns:
            The written file.

        Raises:
            RunFailure: `invalid_model_output` if the features changed
                shape between tiles.
        """
        self._submit(lambda: self._table().close())
        try:
            for future in self._pending:
                _result(future)
        finally:
            self.abort()
        return self._path

    def abort(self) -> None:
        """Drop queued rows, wait for the one being written and release the file."""
        for future in self._pending:
            future.cancel()
        self._thread.shutdown(wait=True)
        if self._writer is not None:
            self._writer.release()

    def _submit(self, work: Callable[[], None]) -> None:
        """Queue work for the writer thread, behind everything queued before."""
        self._pending.append(self._thread.submit(work))

    def _raise_failed(self) -> None:
        """Raise the failure of a finished write, forgetting the ones that succeeded."""
        while self._pending and self._pending[0].done():
            _result(self._pending.pop(0))

    def _table(self) -> embedding_table.TableWriter:
        """The open table; the writer thread created it first."""
        assert self._writer is not None
        return self._writer


def _result(future: Future[None]) -> None:
    """A write's outcome, a row of the wrong shape as `invalid_model_output`."""
    try:
        future.result()
    except embedding_table.RowShapeError as error:
        raise RunFailure(
            code=FailureCode.INVALID_MODEL_OUTPUT,
            safe_message="embedding feature dimensions changed between tiles",
            retryable=False,
        ) from error
