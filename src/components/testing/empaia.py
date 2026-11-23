"""A fake EMPAIA job for testing converters and publishers without the App Service.

`FakeGateway` answers a job's input reads with example data by EAD type and
records every write, so a test sees whether a publisher wrote during
`prepare` (it must not) and during `commit` (it must). `sample_artifact`
writes a small file of a data format for a publisher to read.

A converter for a new EAD type adds example data to `EXAMPLE_INPUTS`; a
publisher for a new data format adds a writer to `SAMPLES`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from components.testing.slides import fake_slide
from model_service.providers.empaia.app_service.gateway import (
    EmpaiaGateway,
    EmpaiaJobMode,
)


if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from rationai.empaia.typing import SlideInfo
    from rationai.empaia.utils.scalar_output import PixelmapTileSet

    from model_service.contracts.json_values import JsonValue
    from model_service.providers.empaia.app_service.gateway import MutationGuard
    from model_service.providers.empaia.app_service.types import DirectAccess


SLIDE = "slide"
SLIDE_SIZE = 256
SLIDE_LEVELS = 4

EXAMPLE_INPUTS: dict[str, JsonValue] = {
    "rectangle": {"upper_left": [32, 32], "width": 64, "height": 64},
    "primitive": {"type": "float", "value": 0.5},
    "string": "text",
}
"""Example input data by EAD type, as the App Service answers it."""


def slide_data() -> dict[str, Any]:
    """The slide input's data, as the App Service answers it.

    Returns:
        A square slide of `SLIDE_SIZE` px with `SLIDE_LEVELS` levels, each
        half the size of the one before.
    """
    return {
        "id": "fake-slide",
        "extent": {"x": SLIDE_SIZE, "y": SLIDE_SIZE, "z": 1},
        "num_levels": SLIDE_LEVELS,
        "pixel_size_nm": {"x": 500.0, "y": 500.0},
        "tile_extent": {"x": 256, "y": 256, "z": 1},
        "levels": [
            {
                "extent": {"x": SLIDE_SIZE // 2**i, "y": SLIDE_SIZE // 2**i, "z": 1},
                "downsample_factor": float(2**i),
            }
            for i in range(SLIDE_LEVELS)
        ],
        "format": "test",
        "tissue": None,
        "stain": None,
    }


class FakeGateway(EmpaiaGateway):
    """One fake EMPAIA job: example inputs in, writes recorded.

    Attributes:
        inputs: Input data by name; names not listed answer by EAD type.
        input_types: The EAD type of each input name, for `EXAMPLE_INPUTS`.
        writes: `(call, name)` for every output posted or pixelmap published.
    """

    def __init__(
        self,
        directory: Path,
        *,
        inputs: dict[str, JsonValue] | None = None,
        input_types: dict[str, str] | None = None,
    ) -> None:
        """Set up the job; its slide is a fake slide file in `directory`.

        Args:
            directory: Where the slide file goes.
            inputs: Input data by name.
            input_types: The EAD type of each input name.
        """
        self._slide_file = fake_slide(directory / "slide.tiff")
        self.inputs = dict(inputs or {})
        self.input_types = dict(input_types or {})
        self.writes: list[tuple[str, str]] = []

    @property
    def job_id(self) -> str:
        """The fake job's ID."""
        return "fake-job"

    def set_before_mutation(self, callback: MutationGuard) -> None:
        """Ignore the mutation guard; nothing here is durable.

        Args:
            callback: The guard.
        """

    async def get_configuration(self) -> dict[str, JsonValue]:
        """Answer an empty configuration.

        Returns:
            `{}`.
        """
        return {}

    async def get_mode(self) -> EmpaiaJobMode:
        """Answer standalone mode.

        Returns:
            `EmpaiaJobMode.STANDALONE`.
        """
        return EmpaiaJobMode.STANDALONE

    async def has_input(self, name: str) -> bool:
        """Every input is present.

        Args:
            name: The input key.

        Returns:
            True.
        """
        return True

    async def get_input_data(self, name: str) -> JsonValue:
        """Answer the named input, the slide's data, or example data by type.

        Args:
            name: The input key.

        Returns:
            The input data.

        Raises:
            KeyError: For an input with neither data nor a type in `EXAMPLE_INPUTS`.
        """
        if name in self.inputs:
            return self.inputs[name]
        if name == SLIDE:
            return slide_data()
        return EXAMPLE_INPUTS[self.input_types.get(name, name)]

    async def direct_access(self, name: str) -> DirectAccess | None:
        """Point every slide input at the fake slide file.

        Args:
            name: The input key.

        Returns:
            The fake slide, with direct access supported.
        """
        from model_service.providers.empaia.app_service.types import DirectAccess

        return DirectAccess(self._slide_file, supported=True)

    async def slide_info(self, name: str) -> SlideInfo:
        """Answer the slide as the SDK types it.

        Args:
            name: The slide's input key.

        Returns:
            The slide of `slide_data()`.
        """
        from rationai.empaia.typing import SlideInfo

        return SlideInfo.model_validate(slide_data())

    async def post_output(self, name: str, body: Any) -> None:
        """Record a posted output.

        Args:
            name: The output key.
            body: The output's body.
        """
        self.writes.append(("post_output", name))

    async def publish_pixelmap(
        self, name: str, slide_id: str, tile_sets: tuple[PixelmapTileSet, ...]
    ) -> None:
        """Record a published pixelmap.

        Args:
            name: The output key.
            slide_id: The slide it lies on.
            tile_sets: Its levels.
        """
        self.writes.append(("publish_pixelmap", name))

    async def finalize(self) -> None:
        """Record finalization."""
        self.writes.append(("finalize", ""))

    async def report_progress(self, fraction: float) -> None:
        """Ignore progress.

        Args:
            fraction: Share of the work done.
        """

    async def report_failure(self, message: str) -> None:
        """Record a failure report.

        Args:
            message: The safe message.
        """
        self.writes.append(("report_failure", message))

    async def close(self) -> None:
        """Nothing to release."""


def _value(path: Path) -> None:
    """Write a `value` file holding one probability.

    Args:
        path: Where to write it.
    """
    from model_service.formats import value

    value.write(path, 0.5)


def _score_grid(path: Path) -> None:
    """Write a one-channel `score_grid` over level 0 of the fake job's slide.

    Args:
        path: Where to write it.
    """
    from model_service.formats import score_grid

    cells = 8
    score_grid.write(
        path,
        score_grid.ScoreGrid(
            values=np.full((1, cells, cells), 0.5, dtype=np.float32),
            scored=np.ones((cells, cells), dtype=bool),
            level=0,
            level_size=(SLIDE_SIZE, SLIDE_SIZE),
            cell_size=SLIDE_SIZE / cells,
            mpp=(0.5, 0.5),
        ),
    )


SAMPLES: dict[str, Callable[[Path], None]] = {
    "value": _value,
    "score_grid": _score_grid,
}
"""A writer of a small file per data format, for publishers to read."""


def sample_artifact(format_name: str, directory: Path) -> tuple[Path, str] | None:
    """Write a small file of a data format.

    Args:
        format_name: The format, e.g. `score_grid`.
        directory: Where to write it.

    Returns:
        The file and its media type, or None if `SAMPLES` has no writer for it.
    """
    from model_service.formats.known import get_format

    write = SAMPLES.get(format_name)
    if write is None:
        return None
    known = get_format(format_name)
    path = directory / f"sample.{known.extension}"
    write(path)
    return path, known.media_type
