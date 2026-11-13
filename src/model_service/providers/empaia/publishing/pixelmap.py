"""Convert builder artifacts and EMPAIA inputs at the EMPAIA image boundary.

Score grids become pixelmaps through the SDK.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from model_service.contracts.result import SharedArtifact
from model_service.contracts.score_grid import read_score_grid
from model_service.providers.empaia.app_service.gateway import EmpaiaGatewayError
from model_service.providers.empaia.app_service.types import SlideLevel
from model_service.providers.empaia.publishing.publishers import OutputPublisher


if TYPE_CHECKING:
    from model_service.contracts.result import BuilderOutput
    from model_service.providers.empaia.app_service.gateway import EmpaiaGateway
    from model_service.providers.empaia.publishing.publishers import Commit


class ScoreGridPublisher(OutputPublisher):
    """Publish the heatmap builder's score grid as an EMPAIA continuous pixelmap.

    The SDK builds the pyramid from the grid: exactly what the model produced,
    in every slide level from the one it was computed at to the coarsest that
    keeps `MIN_PIXELS_PER_SCORE` pixels per score. Tiles are dense, so a
    skipped cell inside an uploaded tile reads 0, but a tile without a scored
    cell is not uploaded.
    """

    def __init__(self, tile_size: int = 512) -> None:
        """Require a provider tile size between 256 and 2048 pixels.

        Args:
            tile_size: The pixelmap tile width and height, in pixels.

        Raises:
            ValueError: If `tile_size` is outside 256 to 2048.
        """
        if not 256 <= tile_size <= 2048:
            raise ValueError("EMPAIA pixelmap tile size must be between 256 and 2048")
        self._tile_size = tile_size

    async def prepare(
        self, gateway: EmpaiaGateway, name: str, output: BuilderOutput
    ) -> Commit:
        """Read the score grid and build its pixelmap pyramid without writing.

        Args:
            gateway: The job's gateway; slide info is read now, the upload happens on
                commit.
            name: The EMPAIA output key, also the pixelmap's name.
            output: The builder output; must be a score grid artifact.

        Returns:
            The commit that uploads the pixelmap.

        Raises:
            EmpaiaGatewayError: Not retryable if the output is not a shared artifact,
                not a readable score grid, has more than one channel, or scores no part
                of the slide; as the gateway raises it if reading slide info fails.
        """
        if not isinstance(output, SharedArtifact):
            raise EmpaiaGatewayError.invalid(
                "EMPAIA shared output must be a score grid"
            )
        path = Path(output.path)
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
        return lambda: gateway.publish_pixelmap(name, slide.id, tile_sets)


MIN_PIXELS_PER_SCORE = 4


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
    levels: tuple[SlideLevel, ...],
    width: int,
    height: int,
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
