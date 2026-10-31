"""Convert builder artifacts and EMPAIA inputs at the EMPAIA image boundary.

Score grids become pixelmaps through the SDK; rectangle masks retain slide
geometry and physical pixel resolution (libvips, loaded lazily); downloaded
slides become tiled TIFFs of one level.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import queue
from collections import deque
from dataclasses import replace
from math import ceil, floor
from pathlib import Path
from typing import TYPE_CHECKING

from model_service.builders.score_grid import MEDIA_TYPE as SCORE_GRID_MEDIA_TYPE
from model_service.builders.score_grid import read_score_grid
from model_service.providers.empaia.gateway import EmpaiaGatewayError
from model_service.providers.empaia.http_gateway import SlideLevel


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import numpy as np
    from numpy.typing import NDArray

    from model_service.builders.contracts import SharedArtifact
    from model_service.providers.empaia.http_gateway import (
        HttpEmpaiaGateway,
        Rectangle,
        SlideMetadata,
    )


class ScoreGridPublisher:
    """Publish the heatmap builder's score grid as an EMPAIA continuous pixelmap.

    The SDK builds the pyramid from the grid: exactly what the model produced,
    in every slide level from the one it was computed at to the coarsest that
    keeps `MIN_PIXELS_PER_SCORE` pixels per score. Tiles are dense, so a
    skipped cell inside an uploaded tile reads 0, but a tile without a scored
    cell is not uploaded.
    """

    def __init__(self, tile_size: int = 512) -> None:
        """Require a provider tile size between 256 and 2048 pixels."""
        if not 256 <= tile_size <= 2048:
            raise ValueError("EMPAIA pixelmap tile size must be between 256 and 2048")
        self._tile_size = tile_size

    async def publish(
        self,
        gateway: HttpEmpaiaGateway,
        name: str,
        artifact: SharedArtifact,
    ) -> None:
        if artifact.media_type != SCORE_GRID_MEDIA_TYPE:
            raise EmpaiaGatewayError.invalid(
                "EMPAIA shared output must be a score grid"
            )
        path = Path(artifact.path)
        try:
            grid = await asyncio.to_thread(read_score_grid, path)
        except (OSError, ValueError, KeyError) as error:
            raise EmpaiaGatewayError.invalid(
                "EMPAIA output artifact is not a readable score grid"
            ) from error
        if grid.values.shape[0] != 1:
            # The SDK uploads single-channel pixelmaps only.
            raise EmpaiaGatewayError.invalid(
                "EMPAIA pixelmaps of more than one channel are not supported"
            )
        if not grid.scored.any():
            raise EmpaiaGatewayError.invalid(
                "the model scored no part of the slide, so there is no heatmap"
            )
        from rationai.empaia.utils.scalar_output import ScalarImageOutput

        slide = await gateway.slide_info()
        levels = tuple(
            SlideLevel(index=index, width=level.extent.x, height=level.extent.y)
            for index, level in enumerate(slide.levels)
        )
        # Matched by size: a downloaded slide's level 0 is one of EMPAIA's levels.
        source = _matching_level(levels, *grid.level_size)
        target = _coarsest_level(levels, source, grid.cell_size)
        scores = ScalarImageOutput(
            values=grid.values[0],
            valid_mask=grid.scored,
            source_level=source.index,
            cell_size=grid.cell_size,  # type: ignore[arg-type]
            legacy_target_level=source.index,
            pixelmap_target_level=target.index,
        )
        tile_sets = await asyncio.to_thread(
            scores.to_pixelmap_pyramid, slide, self._tile_size, self._tile_size
        )
        await gateway.publish_pixelmap(name, slide.id, tile_sets)


class VipsRectangleMaskWriter:
    """Rasterize the intersection of a rectangle and slide into a pyramidal TIFF region mask."""

    def write(
        self,
        destination: Path,
        slide: SlideMetadata,
        rectangle: Rectangle,
    ) -> None:
        """Clip the rectangle to slide bounds and save an 8-bit tiled mask with slide- derived resolution; reject an empty intersection."""
        try:
            import pyvips

            x = min(max(rectangle.x, 0), slide.width)
            y = min(max(rectangle.y, 0), slide.height)
            width = min(rectangle.width, slide.width - x)
            height = min(rectangle.height, slide.height - y)
            if width <= 0 or height <= 0:
                raise EmpaiaGatewayError.invalid(
                    "EMPAIA rectangle input does not overlap the slide"
                )
            mask = pyvips.Image.black(slide.width, slide.height)
            selected = pyvips.Image.black(width, height).new_from_image(255)
            mask = mask.insert(selected, x, y)
            mask.tiffsave(
                str(destination),
                bigtiff=True,
                compression="deflate",
                tile=True,
                tile_width=512,
                tile_height=512,
                xres=1_000_000 / slide.pixel_size_nm_x,
                yres=1_000_000 / slide.pixel_size_nm_y,
                pyramid=True,
            )
        except EmpaiaGatewayError:
            raise
        except Exception as error:
            raise EmpaiaGatewayError.unavailable(
                "EMPAIA rectangle input could not be materialized"
            ) from error


class TiffSlideDownloader:
    """Download the slide level a builder reads into a tiled JPEG TIFF.

    Regions are fetched tile by tile, a few at a time and in order, and streamed
    into the file, so memory stays bounded whatever the slide size. The TIFF
    records the level's resolution, so OpenSlide reports its mpp and the builder
    reads it as level 0.
    """

    def __init__(
        self, tile_size: int = 512, concurrency: int = 8, quality: int = 90
    ) -> None:
        self._tile_size = tile_size
        self._concurrency = concurrency
        self._quality = quality

    async def download(
        self,
        gateway: HttpEmpaiaGateway,
        slide: SlideMetadata,
        mpp: float,
        destination: Path,
        *,
        region: Rectangle | None = None,
        margin: int = 0,
    ) -> None:
        """Fetch the tiles of the level closest to `mpp` and write them to `destination`.

        With a `region`, tiles farther than `margin` level pixels from it are not
        fetched but written white, which the builder treats as background.
        """
        level = closest_level(slide, mpp)
        size = self._tile_size
        positions = [
            (x, y)
            for y in range(0, level.height, size)
            for x in range(0, level.width, size)
        ]
        wanted = _within(region, margin, slide, level, size)
        logger.info(
            "downloading slide %s level %d (%dx%d px, %d of %d tiles)",
            slide.identifier,
            level.index,
            level.width,
            level.height,
            sum(wanted(x, y) for x, y in positions),
            len(positions),
        )

        async def fetch(x: int, y: int) -> NDArray[np.uint8] | None:
            if not wanted(x, y):
                return None
            return await gateway.read_region(
                slide.identifier,
                level.index,
                x,
                y,
                min(size, level.width - x),
                min(size, level.height - y),
            )

        tiles: queue.Queue[object] = queue.Queue(maxsize=2 * self._concurrency)
        writer = asyncio.ensure_future(
            asyncio.to_thread(self._write, destination, slide, level, tiles)
        )
        pending: deque[tuple[int, int, asyncio.Task[NDArray[np.uint8] | None]]] = (
            deque()
        )
        remaining = iter(positions)
        try:
            for x, y in itertools.islice(remaining, self._concurrency):
                pending.append((x, y, asyncio.ensure_future(fetch(x, y))))
            while pending:
                x, y, task = pending.popleft()
                data = await task
                for nx, ny in itertools.islice(remaining, 1):
                    pending.append((nx, ny, asyncio.ensure_future(fetch(nx, ny))))
                if data is None:
                    tile = _blank_tile(size)
                else:
                    tile = await asyncio.to_thread(
                        _pad_tile,
                        data,
                        min(size, level.width - x),
                        min(size, level.height - y),
                        size,
                    )
                await _hand_over(tiles, tile, writer)
            await writer
            await self._append_overview(gateway, slide, level, destination)
        except BaseException:
            for _, _, task in pending:
                task.cancel()
            if not writer.done():
                await _hand_over(tiles, _ABORT, writer)
            with contextlib.suppress(BaseException):
                await writer
            destination.unlink(missing_ok=True)
            raise

    async def _append_overview(
        self,
        gateway: HttpEmpaiaGateway,
        slide: SlideMetadata,
        level: SlideLevel,
        destination: Path,
    ) -> None:
        """Append a coarse level, so builders find tissue without reading the full level."""
        import numpy as np

        from model_service.builders.tissue import MAX_PIXELS, TISSUE_MPP

        coarser = tuple(other for other in slide.levels if other.width < level.width)
        if not coarser:
            return
        overview = closest_level(replace(slide, levels=coarser), TISSUE_MPP)
        if overview.width * overview.height > MAX_PIXELS:
            return
        size = self._tile_size
        pixels = np.zeros((overview.height, overview.width, 3), dtype=np.uint8)
        limit = asyncio.Semaphore(self._concurrency)

        async def fetch(x: int, y: int) -> None:
            width = min(size, overview.width - x)
            height = min(size, overview.height - y)
            async with limit:
                data = await gateway.read_region(
                    slide.identifier, overview.index, x, y, width, height
                )
            tile = await asyncio.to_thread(_pad_tile, data, width, height, size)
            pixels[y : y + height, x : x + width] = tile[:height, :width]

        await asyncio.gather(
            *(
                fetch(x, y)
                for y in range(0, overview.height, size)
                for x in range(0, overview.width, size)
            )
        )
        await asyncio.to_thread(self._write_page, destination, slide, overview, pixels)

    def _write_page(
        self,
        destination: Path,
        slide: SlideMetadata,
        level: SlideLevel,
        pixels: NDArray[np.uint8],
    ) -> None:
        import tifffile

        mpp_x = slide.pixel_size_nm_x / 1000 * slide.width / level.width
        mpp_y = slide.pixel_size_nm_y / 1000 * slide.height / level.height
        try:
            tifffile.imwrite(
                destination,
                pixels,
                append=True,
                subfiletype=1,  # reduced resolution: OpenSlide reads it as a level
                tile=(256, 256),
                photometric="rgb",
                compression="jpeg",
                compressionargs={"level": self._quality},
                resolution=(10_000 / mpp_x, 10_000 / mpp_y),
                resolutionunit="CENTIMETER",
                bigtiff=True,
            )
        except Exception as error:
            raise EmpaiaGatewayError.unavailable(
                "EMPAIA slide could not be written"
            ) from error

    def _write(
        self,
        destination: Path,
        slide: SlideMetadata,
        level: SlideLevel,
        tiles: queue.Queue[object],
    ) -> None:
        import tifffile

        def received() -> Iterator[NDArray[np.uint8]]:
            while True:
                tile = tiles.get()
                if tile is _ABORT:
                    raise EmpaiaGatewayError.unavailable(
                        "EMPAIA slide download was interrupted"
                    )
                yield tile  # type: ignore[misc]

        # OpenSlide derives mpp from the resolution in pixels per centimetre.
        mpp_x = slide.pixel_size_nm_x / 1000 * slide.width / level.width
        mpp_y = slide.pixel_size_nm_y / 1000 * slide.height / level.height
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            tifffile.imwrite(
                destination,
                data=received(),
                shape=(level.height, level.width, 3),
                dtype="uint8",
                tile=(self._tile_size, self._tile_size),
                photometric="rgb",
                compression="jpeg",
                compressionargs={"level": self._quality},
                resolution=(10_000 / mpp_x, 10_000 / mpp_y),
                resolutionunit="CENTIMETER",
                bigtiff=True,
            )
        except EmpaiaGatewayError:
            raise
        except Exception as error:
            raise EmpaiaGatewayError.unavailable(
                "EMPAIA slide could not be written"
            ) from error


_ABORT = object()


def _within(
    region: Rectangle | None,
    margin: int,
    slide: SlideMetadata,
    level: SlideLevel,
    size: int,
) -> Callable[[int, int], bool]:
    """Whether a level tile at (x, y) lies within `margin` of `region`."""
    if region is None:
        return lambda _x, _y: True
    scale_x = level.width / slide.width
    scale_y = level.height / slide.height
    left = floor(region.x * scale_x) - margin
    top = floor(region.y * scale_y) - margin
    right = ceil((region.x + region.width) * scale_x) + margin
    bottom = ceil((region.y + region.height) * scale_y) + margin
    return lambda x, y: x < right and x + size > left and y < bottom and y + size > top


def _blank_tile(size: int) -> NDArray[np.uint8]:
    import numpy as np

    return np.full((size, size, 3), 255, dtype=np.uint8)


async def _hand_over(
    tiles: queue.Queue[object], item: object, writer: asyncio.Future[None]
) -> None:
    """Queue an item for the writer thread without blocking forever if it stopped."""
    while True:
        if writer.done():
            writer.result()
            return
        try:
            tiles.put_nowait(item)
            return
        except queue.Full:
            await asyncio.sleep(0.05)


def _pad_tile(
    pixels: NDArray[np.uint8], width: int, height: int, tile_size: int
) -> NDArray[np.uint8]:
    """Turn a region's pixels into an RGB tile, padded to the full tile size."""
    import numpy as np

    pixels = np.asarray(pixels)
    if pixels.ndim == 2:
        pixels = np.repeat(pixels[..., None], 3, axis=2)
    if pixels.ndim != 3 or pixels.shape[2] < 3 or pixels.shape[:2] != (height, width):
        raise EmpaiaGatewayError.invalid(
            "EMPAIA slide region has unexpected dimensions"
        )
    tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
    tile[:height, :width] = pixels[..., :3]
    return tile


def closest_level(slide: SlideMetadata, mpp: float) -> SlideLevel:
    """Choose the level OpenSlide's `closest_level` would choose on the original slide."""
    scale = (
        mpp / (slide.pixel_size_nm_x / 1000) + mpp / (slide.pixel_size_nm_y / 1000)
    ) / 2
    return min(
        slide.levels,
        key=lambda level: abs(
            (slide.width / level.width + slide.height / level.height) / 2 - scale
        ),
    )


MIN_PIXELS_PER_SCORE = 4


def _coarsest_level(
    levels: tuple[SlideLevel, ...], level: SlideLevel, spacing: float
) -> SlideLevel:
    """The coarsest level at least `MIN_PIXELS_PER_SCORE` pixels per score."""
    best = level
    for candidate in levels:
        downsample = max(level.width / candidate.width, level.height / candidate.height)
        if (
            candidate.width < best.width
            and spacing / downsample >= MIN_PIXELS_PER_SCORE
        ):
            best = candidate
    return best


def _matching_level(
    levels: tuple[SlideLevel, ...],
    width: int,
    height: int,
) -> SlideLevel:
    """Choose the pyramid level minimizing the total absolute width and height mismatch."""
    return min(
        levels,
        key=lambda level: abs(level.width - width) + abs(level.height - height),
    )
