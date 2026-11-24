"""`continuous_pixelmap`: a one-channel score grid as an EMPAIA continuous pixelmap."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import Field

from components.base import Commit, EmpaiaOutput, Problem, Publisher, PublisherOptions
from model_service.formats import score_grid
from model_service.providers.empaia.app_service.gateway import EmpaiaGatewayError
from model_service.providers.empaia.app_service.types import SlideLevel


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from components.base import Col
    from model_service.providers.empaia.app_service.gateway import EmpaiaGateway


MIN_PIXELS_PER_SCORE = 4


class ContinuousPixelmapOptions(PublisherOptions):
    """Options of `continuous_pixelmap`."""

    tile_size: int = Field(
        512, ge=256, le=2048, description="Pixelmap tile edge in pixels.", strict=True
    )


class ContinuousPixelmap(Publisher):
    """Publishes a one-channel score grid as a continuous pixelmap on the app's slide.

    The SDK builds the pyramid from the grid: exactly what the model produced,
    in every slide level from the one it was computed at to the coarsest that
    keeps `MIN_PIXELS_PER_SCORE` pixels per score. Tiles are dense, so an
    unscored cell inside an uploaded tile reads 0, but a tile without a
    scored cell is not uploaded. EMPAIA pixelmaps carry one channel here, so
    a multi-channel grid is refused when the analysis loads.
    """

    name = "continuous_pixelmap"
    Options = ContinuousPixelmapOptions
    reads_format = score_grid.FORMAT.name
    emits = frozenset({"continuous_pixelmap"})
    example_options: ClassVar[dict[str, Any]] = {"type": "continuous_pixelmap"}
    options: ContinuousPixelmapOptions

    def io_fragment(self, item: EmpaiaOutput) -> dict[str, Any]:
        """Declare a continuous pixelmap on the app's slide.

        Args:
            item: The output; `slide` names the app's slide.

        Returns:
            The pixelmap entry, referencing the slide.
        """
        return {"type": "continuous_pixelmap", "reference": f"io.{item.slide}"}

    def check(self, columns: Mapping[str, Col]) -> Problem | None:
        """Refuse a grid of more than one channel.

        Args:
            columns: The columns reaching the sinks; `scores` gives the channels.

        Returns:
            The problem, or None for one channel.
        """
        shape = columns["scores"].shape if "scores" in columns else None
        channels = shape[0] if shape else None
        if channels == 1:
            return None
        return Problem(
            "channels",
            f"a continuous pixelmap carries one channel; the scores have {channels}",
            "A model step whose scores have one channel",
            {"channels": channels},
        )

    async def prepare(
        self,
        gateway: EmpaiaGateway,
        item: EmpaiaOutput,
        artifact: Path,
        media_type: str,
    ) -> Commit:
        """Read the score grid and build its pixelmap pyramid without writing.

        Args:
            gateway: The job's gateway; slide info is read now, the upload
                happens on commit.
            item: The output; also the pixelmap's name.
            artifact: The stored score grid.
            media_type: Its stored media type.

        Returns:
            The commit that uploads the pixelmap.

        Raises:
            EmpaiaGatewayError: Not retryable if the file is not a readable
                score grid, has more than one channel, or scores no part of the
                slide; as the gateway raises it if reading slide info fails.
        """
        try:
            grid = await asyncio.to_thread(score_grid.read, artifact, media_type)
        except (OSError, ValueError, KeyError) as error:
            raise EmpaiaGatewayError.invalid(
                "EMPAIA output artifact is not a readable score grid"
            ) from error
        if grid.values.shape[0] != 1:
            raise EmpaiaGatewayError.invalid(
                "EMPAIA pixelmaps of more than one channel are not supported"
            )
        if not grid.scored.any():
            raise EmpaiaGatewayError.invalid(
                "the model scored no part of the slide, so there is no heatmap"
            )
        from rationai.empaia.utils.scalar_output import ScalarImageOutput

        slide = await gateway.slide_info(item.slide or "slide")
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
        size = self.options.tile_size
        tile_sets = await asyncio.to_thread(
            scores.to_pixelmap_pyramid, slide, size, size
        )

        async def commit() -> None:
            """Upload the prepared pixelmap pyramid."""
            await gateway.publish_pixelmap(item.name, slide.id, tile_sets)

        return commit


def _coarsest_level(
    levels: tuple[SlideLevel, ...], level: SlideLevel, spacing: float
) -> SlideLevel:
    """Pick the coarsest level that keeps at least `MIN_PIXELS_PER_SCORE` pixels per score.

    Args:
        levels: The slide's pyramid levels.
        level: The level the scores were computed at.
        spacing: The score cell size, in pixels of `level`.

    Returns:
        The coarsest qualifying level, or `level` if none coarser qualifies.
    """
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
    levels: tuple[SlideLevel, ...], width: int, height: int
) -> SlideLevel:
    """Choose the pyramid level minimizing the total absolute width and height mismatch.

    Args:
        levels: The slide's pyramid levels.
        width: The width to match.
        height: The height to match.

    Returns:
        The closest level.
    """
    return min(
        levels,
        key=lambda level: abs(level.width - width) + abs(level.height - height),
    )
