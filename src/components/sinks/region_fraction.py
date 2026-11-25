"""`region_fraction`: per region, the share of scored pixels at or above a threshold.

`model_service.slides.regions` cuts each region to each tile with ratiopath's
`tile_annotations` and finds the score map's pixels inside with rasterio.
Regions come from the source's frame, in level-0 pixels.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, ClassVar

from pydantic import Field

from components import Col, PerRun, Sink, SinkOptions
from model_service.formats import region_values


if TYPE_CHECKING:
    from pathlib import Path

    from shapely.geometry.base import BaseGeometry

    from components import Batch, Run


WHOLE_SLIDE = "slide"


class RegionFractionOptions(SinkOptions):
    """Options of `region_fraction`."""

    threshold: PerRun[
        Annotated[
            float,
            Field(
                ge=0,
                le=1,
                allow_inf_nan=False,
                description="The score at or above which a pixel counts, from 0 to 1.",
            ),
        ]
    ] = 0.5


class RegionFraction(Sink):
    """The share of each region's scored pixels whose score reaches the threshold.

    Writes `<output>.json` in the `region_values` format, one value per region
    in the regions' order. A pixel counts when its centre lies in the region,
    and only tiles the source read are scored, so background the source
    skipped is left out of both counts. Where tiles overlap, a pixel counts
    once per tile, so the share is weighted by overlap, evenly inside the
    grid. Without regions the whole slide is one region, `slide`.
    """

    name = "region_fraction"
    Options = RegionFractionOptions
    format = region_values.FORMAT.name
    example_options: ClassVar[dict[str, Any]] = {"output": "purity"}
    reads: ClassVar[dict[str, Col]] = {
        "tile_x": Col("int64"),
        "tile_y": Col("int64"),
        "scores": Col("float32", (1, None, None)),
    }
    frame_reads = frozenset({"regions", "tile_extent_x", "tile_extent_y", "downsample"})
    options: RegionFractionOptions

    def open(self, frame: dict[str, Any], run: Run) -> None:
        """Take the regions and tile geometry from the frame.

        Args:
            frame: The source's frame row.
            run: The run's `Run`.
        """
        regions = frame.get("regions")
        self._ids = [WHOLE_SLIDE] if regions is None else [r.id for r in regions]
        self._shapes: list[BaseGeometry] | None = (
            None if regions is None else [r.geometry for r in regions]
        )
        self._tile = (int(frame["tile_extent_x"]), int(frame["tile_extent_y"]))
        self._downsample = float(frame["downsample"])
        self._inside = [0] * len(self._ids)
        self._positive = [0] * len(self._ids)
        self._path: Path = (
            run.work_dir / f"{self.options.output}.{region_values.FORMAT.extension}"
        )

    def add(self, batch: Batch) -> None:
        """Count each tile's pixels in each region it overlaps.

        Args:
            batch: `tile_x`, `tile_y` and `scores`.
        """
        from model_service.slides.regions import tile_part, tile_pixels

        for x, y, scores in zip(
            batch["tile_x"], batch["tile_y"], batch["scores"], strict=True
        ):
            positive = scores[0] >= self.options.threshold
            if self._shapes is None:
                self._inside[0] += positive.size
                self._positive[0] += int(positive.sum())
                continue
            for index, shape in enumerate(self._shapes):
                part = tile_part([shape], int(x), int(y), self._tile, self._downsample)
                if part.area <= 0:
                    continue
                inside = tile_pixels(part, int(x), int(y), self._tile, positive.shape)
                self._inside[index] += int(inside.sum())
                self._positive[index] += int((positive & inside).sum())

    def close(self) -> Path:
        """Write each region's share; a region without scored pixels gets none.

        Returns:
            The written file.
        """
        region_values.write(
            self._path,
            tuple(
                region_values.RegionValue(
                    region_id, positive / inside if inside else None
                )
                for region_id, inside, positive in zip(
                    self._ids, self._inside, self._positive, strict=True
                )
            ),
        )
        return self._path
