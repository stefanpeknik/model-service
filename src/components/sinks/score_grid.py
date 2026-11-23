"""`score_grid`: averages per-tile score maps into a `score_grid` file.

Ratiopath's memory-mapped `MaskBuilder` averages overlapping scores. The
grid's geometry comes from the source's frame; its channels and per-tile map
size from the `scores` column's shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from components.base import Col, Sink
from model_service.formats import score_grid


if TYPE_CHECKING:
    from pathlib import Path

    from components.base import Batch
    from components.base.run import Run


class ScoreGridSink(Sink):
    """Averages each tile's `scores` over the cells the tile covers into a score grid.

    Writes `<output>.npz` in the `score_grid` format: float32 scores in 0..1
    of shape (channels, rows, columns), with which cells any tile covered.
    """

    name = "score_grid"
    format = score_grid.NAME
    example_options: ClassVar[dict[str, Any]] = {"output": "heatmap"}
    reads: ClassVar[dict[str, Col]] = {
        "tile_x": Col("int64"),
        "tile_y": Col("int64"),
        "scores": Col("float32", (None, None, None)),
    }
    frame_reads = frozenset(
        {
            "extent_x",
            "extent_y",
            "tile_extent_x",
            "tile_extent_y",
            "stride_x",
            "stride_y",
            "level",
            "mpp_x",
            "mpp_y",
        }
    )

    def open(self, frame: dict[str, Any], run: Run) -> None:
        """Size the grid from the frame and the `scores` shape.

        Args:
            frame: The source's frame row.
            run: The run's `Run`.

        Raises:
            ValueError: If the `scores` shape is not fully known.
        """
        from ratiopath.masks.mask_builders import MaskBuilder

        shape = run.columns["scores"].shape
        if shape is None or not all(isinstance(d, int) for d in shape):
            raise ValueError(f"score_grid needs a known scores shape, got {shape}")
        channels, height, width = shape
        self._frame = frame
        self._path: Path = (
            run.work_dir / f"{self.options.output}.{score_grid.EXTENSION}"
        )
        self._mask = MaskBuilder(
            source_extents=(frame["extent_y"], frame["extent_x"]),
            source_tile_extent=(frame["tile_extent_y"], frame["tile_extent_x"]),
            output_tile_extent=(height, width),
            stride=(frame["stride_y"], frame["stride_x"]),
            n_channels=channels,
            storage="memmap",
        )

    def add(self, batch: Batch) -> None:
        """Add a batch of tiles' scores at their positions.

        Args:
            batch: `tile_x`, `tile_y` and `scores`.
        """
        import numpy as np

        coords = np.stack([batch["tile_y"], batch["tile_x"]], axis=1)
        self._mask.update_batch(batch=batch["scores"], coords=coords)

    def close(self) -> Path:
        """Average and write the grid; the memory map is removed either way.

        Returns:
            The written file.
        """
        import numpy as np

        try:
            finalized = self._mask.finalize()
            values = np.asarray(finalized["mask"], dtype=np.float32)
            counts = np.asarray(finalized["overlap_counter"])
            # The grid covers the padded span, so a cell is span / cells
            # pixels of the level along each axis.
            scale_y, scale_x = self._mask.span / np.asarray(values.shape[1:])
            frame = self._frame
            score_grid.write(
                self._path,
                score_grid.ScoreGrid(
                    values=values,
                    scored=counts.reshape(values.shape[1:]) > 0,
                    level=frame["level"],
                    level_size=(frame["extent_x"], frame["extent_y"]),
                    cell_size=float(min(scale_x, scale_y)),
                    mpp=(float(frame["mpp_x"]), float(frame["mpp_y"])),
                ),
            )
        finally:
            self._mask.cleanup()
        return self._path

    def abort(self) -> None:
        """Remove the memory map of a failed run."""
        self._mask.cleanup()
