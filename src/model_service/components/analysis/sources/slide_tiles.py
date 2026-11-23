"""`slide_tiles`: grid tiles of the slide level closest to the model's resolution.

The geometry is the team's code: `ratiopath.tiling.grid_tiles` places the
tiles, and `model_service.slides.tile_reader.TileReader` picks the level and drops tiles
outside the tissue or the region of interest. Frame fields follow ratiopath's
slide rows; `tile` is channels-first, as served models take it.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal

from pydantic import Field

from model_service.components import Col, Opened, Options, PerRun, Problem, Source


if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

    import numpy as np

    from model_service.components import Batch
    from model_service.components.run import Run


SLIDE = "slide"
REGION_OF_INTEREST = "region_of_interest"

# The per-run settings. Their descriptions are what users read: in the public
# API's settings schema, and in EMPAIA's job form when an app offers them.
StrideFraction = Annotated[
    float,
    Field(
        gt=0,
        le=1,
        allow_inf_nan=False,
        description=(
            "How far apart tiles are, as a share of the tile size: above 0, up "
            "to 1. 1 places tiles side by side; 0.5 overlaps them by half, for "
            "a finer result that takes about four times as long."
        ),
    ),
]
SkipBackground = Annotated[
    bool,
    Field(
        description=(
            "Whether tiles without tissue are skipped; they get no result. "
            "Skipping is faster and leaves empty glass out. On by default."
        )
    ),
]
TissueFraction = Annotated[
    float,
    Field(
        ge=0,
        lt=1,
        allow_inf_nan=False,
        description=(
            "How much of a tile must be tissue for it to be analysed, when "
            "background is skipped: from 0 up to but not including 1. 0 "
            "analyses every tile with any tissue; 0.5 needs at least half."
        ),
    ),
]


class SlideTilesOptions(Options):
    """Options of `slide_tiles`."""

    tile_size: int | None = Field(
        None,
        gt=0,
        description="Tile edge in pixels; the first model step's when left out.",
    )
    mpp: float | None = Field(
        None,
        gt=0,
        allow_inf_nan=False,
        description=(
            "Microns per pixel to read at (the closest slide level); the first "
            "model step's when left out."
        ),
    )
    region_of_interest: Literal["required", "optional", "ignored"] = Field(
        "optional",
        description=(
            "Whether a run must give a region-of-interest mask (`required`), "
            "may (`optional`) or may not (`ignored`); only tiles the mask "
            "covers are read."
        ),
    )
    stride_fraction: PerRun[StrideFraction] = 0.5
    skip_background: PerRun[SkipBackground] = True
    min_tissue_fraction: PerRun[TissueFraction] = 0.0


class SlideTiles(Source):
    """Grid tiles of one slide level, without background or tiles outside the ROI.

    Every grid position is a key, so skipped tiles still count toward
    progress. The grid covers at least one tile, even on a slide smaller
    than a tile.
    """

    name = "slide_tiles"
    Options = SlideTilesOptions
    example_options: ClassVar[dict[str, Any]] = {
        "tile_size": 64,
        "mpp": 0.5,
        "skip_background": False,
    }
    adds: ClassVar[dict[str, Col]] = {
        "tile_x": Col("int64"),
        "tile_y": Col("int64"),
        "tile": Col("uint8", (3, "tile_size", "tile_size")),
    }
    frame: ClassVar[dict[str, Col]] = {
        "extent_x": Col("int64"),
        "extent_y": Col("int64"),
        "tile_extent_x": Col("int64"),
        "tile_extent_y": Col("int64"),
        "stride_x": Col("int64"),
        "stride_y": Col("int64"),
        "level": Col("int64"),
        "mpp_x": Col("float64"),
        "mpp_y": Col("float64"),
        "downsample": Col("float64"),
    }
    options: SlideTilesOptions

    def inputs(self) -> dict[str, bool]:
        """The slide, and the region of interest unless it is ignored.

        Returns:
            Input name to whether it is required.
        """
        roi = self.options.region_of_interest
        if roi == "ignored":
            return {SLIDE: True}
        return {SLIDE: True, REGION_OF_INTEREST: roi == "required"}

    @classmethod
    def hint(cls, problem: Problem) -> str | None:
        """Teach the geometry options.

        Args:
            problem: The failed check.

        Returns:
            How to fix a geometry mistake, or None.
        """
        if problem.detail.get("option") in {"tile_size", "mpp"}:
            return (
                "leave `tile_size` and `mpp` out so the first model step sets "
                "them; a model-free analysis sets both"
            )
        return None

    @contextmanager
    def open(self, inputs: dict[str, Path], run: Run) -> Iterator[Opened]:
        """Open the slide (and mask), find tissue, and lay out the grid.

        Args:
            inputs: `slide`, and `region_of_interest` if the run gives one.
            run: The run's `Run`; unused.

        Yields:
            The frame, one key `(x, y)` per grid position, and the loader.

        Raises:
            InvalidInvocation: The slide or mask is unreadable, or tissue
                detection cannot run on the slide.
            ValueError: If `tile_size` or `mpp` was never filled in.
        """
        import numpy as np
        from ratiopath.tiling import grid_tiles

        from model_service.slides.tile_reader import TileReader, open_slide

        tile_size, mpp = self.options.tile_size, self.options.mpp
        if tile_size is None or mpp is None:
            raise ValueError("slide_tiles needs tile_size and mpp before it opens")
        stride = max(1, round(self.options.stride_fraction * tile_size))
        roi = (
            inputs.get(REGION_OF_INTEREST)
            if self.options.region_of_interest != "ignored"
            else None
        )
        with (
            open_slide(inputs[SLIDE], SLIDE) as slide,
            TileReader.open(
                slide,
                roi,
                mpp,
                skip_background=self.options.skip_background,
                min_tissue_fraction=self.options.min_tissue_fraction,
            ) as reader,
        ):
            extent_x, extent_y = slide.level_dimensions[reader.level]
            mpp_x, mpp_y = slide.slide_resolution(reader.level)
            frame = {
                "extent_x": extent_x,
                "extent_y": extent_y,
                "tile_extent_x": tile_size,
                "tile_extent_y": tile_size,
                "stride_x": stride,
                "stride_y": stride,
                "level": reader.level,
                "mpp_x": float(mpp_x),
                "mpp_y": float(mpp_y),
                "downsample": float(slide.level_downsamples[reader.level]),
            }
            keys = list(
                grid_tiles(
                    slide_extent=(max(extent_x, tile_size), max(extent_y, tile_size)),
                    tile_extent=(tile_size, tile_size),
                    stride=(stride, stride),
                )
            )

            def load(chosen: Sequence[Any]) -> Batch:
                """Read the chosen tiles, dropping the ones the reader skips.

                Args:
                    chosen: Tile corners `(x, y)` on the level read.

                Returns:
                    `tile_x`, `tile_y` and `tile` of the tiles kept.
                """
                read = [(x, y, reader.read(x, y, tile_size)) for x, y in chosen]
                kept = [(x, y, tile) for x, y, tile in read if tile is not None]
                tiles: np.ndarray = (
                    np.stack([tile for _, _, tile in kept])
                    if kept
                    else np.empty((0, 3, tile_size, tile_size), np.uint8)
                )
                return {
                    "tile_x": np.array([x for x, _, _ in kept], np.int64),
                    "tile_y": np.array([y for _, y, _ in kept], np.int64),
                    "tile": tiles,
                }

            yield Opened(frame=frame, keys=keys, load=load)
