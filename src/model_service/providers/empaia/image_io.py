"""Convert local TIFF artifacts and rectangle inputs at the EMPAIA image boundary.

libvips is loaded lazily. TIFF heatmaps become float32 pixelmap tiles; rectangle masks
retain slide geometry and physical pixel resolution; downloaded slides become tiled
TIFFs of one level.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import itertools
import json
import logging
import math
import queue
from collections import deque
from dataclasses import replace
from math import ceil, floor
from pathlib import Path
from typing import TYPE_CHECKING

from model_service.builders.contracts import SCORE_SPACING_KEY
from model_service.providers.empaia.gateway import EmpaiaGatewayError


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    import numpy as np
    from numpy.typing import NDArray

    from model_service.builders.contracts import SharedArtifact
    from model_service.providers.empaia.http_gateway import (
        HttpEmpaiaGateway,
        Rectangle,
        SlideLevel,
        SlideMetadata,
    )


class TiffPixelmapPublisher:
    """Publish a score TIFF as an EMPAIA float32 continuous pixelmap in [0, 1].

    8-bit TIFFs (the heatmap builder's default) are scaled from 0-255; float32
    TIFFs are published unchanged. Tiles are sent channel-major.

    The heatmap builder upsamples its score grid to the slide level it read, and
    records how many pixels one score spans in a pyramidal TIFF. The pixelmap
    goes to the coarsest slide level that still keeps `MIN_PIXELS_PER_SCORE`
    pixels per score, so no score is lost while far fewer tiles are uploaded.
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
        """Validate and read the TIFF, choose the nearest slide level, create a pixelmap, and upload padded float32 tiles sequentially."""
        if artifact.media_type != "image/tiff":
            raise EmpaiaGatewayError.invalid(
                "EMPAIA shared output must be an image/tiff artifact"
            )
        path = Path(artifact.path)
        if not path.is_file():
            raise EmpaiaGatewayError.invalid("EMPAIA output artifact is unavailable")
        try:
            import pyvips

            image = pyvips.Image.new_from_file(str(path), access="sequential")
        except Exception as error:
            raise EmpaiaGatewayError.invalid(
                "EMPAIA output artifact is not a readable TIFF"
            ) from error
        if image.format not in {"uchar", "float"}:
            raise EmpaiaGatewayError.invalid(
                "EMPAIA score TIFF must contain uint8 or float32 values"
            )
        scale = 255.0 if image.format == "uchar" else 1.0
        slide = await gateway.slide_metadata()
        level = _matching_level(slide.levels, image.width, image.height)
        spacing = _score_spacing(image)
        if spacing is not None:
            target = _coarsest_level(slide.levels, level, spacing)
            if target != level:
                # Loading by file starts from the builder's nearest pyramid page
                # instead of shrinking the full-resolution page.
                image = pyvips.Image.thumbnail(
                    str(path), target.width, height=target.height, size="force"
                )
                level = target
        tiles_x = math.ceil(image.width / self._tile_size)
        tiles_y = math.ceil(image.height / self._tile_size)
        pixelmap_id = await gateway.create_pixelmap(
            name,
            {
                "name": name,
                "type": "continuous_pixelmap",
                "element_type": "float32",
                "channel_count": image.bands,
                "tilesize": self._tile_size,
                "min_value": 0.0,
                "max_value": 1.0,
                "reference_id": slide.identifier,
                "reference_type": "wsi",
                "creator_type": "job",
                "creator_id": gateway.job_id,
                "levels": [
                    {
                        "slide_level": level.index,
                        "position_min_x": 0,
                        "position_max_x": tiles_x - 1,
                        "position_min_y": 0,
                        "position_max_y": tiles_y - 1,
                    }
                ],
            },
        )
        for tile_y in range(tiles_y):
            for tile_x in range(tiles_x):
                x = tile_x * self._tile_size
                y = tile_y * self._tile_size
                width = min(self._tile_size, image.width - x)
                height = min(self._tile_size, image.height - y)
                tile = image.crop(x, y, width, height)
                if width != self._tile_size or height != self._tile_size:
                    tile = tile.embed(
                        0,
                        0,
                        self._tile_size,
                        self._tile_size,
                        extend="black",
                    )
                await gateway.put_pixelmap_tile(
                    pixelmap_id,
                    level.index,
                    tile_x,
                    tile_y,
                    _channel_major(tile.cast("float") / scale),
                )


def _channel_major(tile: object) -> bytes:
    """Libvips stores bands interleaved; EMPAIA expects one plane per channel."""
    if tile.bands == 1:  # type: ignore[attr-defined]
        return bytes(tile.write_to_memory())  # type: ignore[attr-defined]
    # Render once: the source is opened for sequential access.
    rendered = tile.copy_memory()  # type: ignore[attr-defined]
    return b"".join(bytes(band.write_to_memory()) for band in rendered.bandsplit())


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

        async def fetch(x: int, y: int) -> bytes | None:
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
        pending: deque[tuple[int, int, asyncio.Task[bytes | None]]] = deque()
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
                        _decode_tile,
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
            tile = await asyncio.to_thread(_decode_tile, data, width, height, size)
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


def _decode_tile(
    data: bytes, width: int, height: int, tile_size: int
) -> NDArray[np.uint8]:
    """Decode a region image to an RGB tile, padded to the full tile size."""
    import numpy as np
    from PIL import Image

    try:
        with Image.open(io.BytesIO(data)) as image:
            pixels = np.asarray(image.convert("RGB"))
    except Exception as error:
        raise EmpaiaGatewayError.invalid(
            "EMPAIA slide region is not a readable image"
        ) from error
    if pixels.shape[:2] != (height, width):
        raise EmpaiaGatewayError.invalid(
            "EMPAIA slide region has unexpected dimensions"
        )
    tile = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
    tile[:height, :width] = pixels
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


def _score_spacing(image: object) -> float | None:
    """Pixels per score recorded by the heatmap builder, if any."""
    try:
        description = json.loads(image.get("image-description"))  # type: ignore[attr-defined]
        spacing = float(description[SCORE_SPACING_KEY])
    except Exception:
        return None
    return spacing if math.isfinite(spacing) and spacing > 0 else None


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
