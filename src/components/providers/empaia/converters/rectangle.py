"""`rectangle`: a region of interest drawn on the slide, as a mask the size of the slide."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from components import EmpaiaInput, InputConverter


if TYPE_CHECKING:
    from pathlib import Path

    from model_service.providers.empaia.app_service.gateway import EmpaiaGateway


ROI_CLASS = "org.empaia.global.v1.classes.roi"
MASK_FILE = "mask.tiff"


class Rectangle(InputConverter):
    """A rectangle on the app's slide, given to the pipeline as a region-of-interest mask.

    The rectangle, clipped to the slide, becomes a pyramidal 8-bit TIFF of
    the slide's size and pixel size: 255 inside, 0 outside. EMPAIA offers it
    with the region-of-interest class, so the viewer lets a user draw it.
    """

    name = "rectangle"
    ead_type = "rectangle"

    def io_fragment(self, item: EmpaiaInput) -> dict[str, Any]:
        """Declare a region-of-interest rectangle on the app's slide.

        Args:
            item: The input; `slide` names the app's slide.

        Returns:
            The rectangle entry, referencing the slide.
        """
        return {
            "type": self.ead_type,
            "reference": f"io.{item.slide}",
            "classes": [ROI_CLASS],
        }

    async def materialize(
        self, gateway: EmpaiaGateway, item: EmpaiaInput, directory: Path
    ) -> Path:
        """Read the rectangle and its slide, and write the mask into `directory`.

        Args:
            gateway: The job's gateway.
            item: The input; `slide` names the slide it lies on.
            directory: Where to write `mask.tiff`.

        Returns:
            The mask.

        Raises:
            EmpaiaGatewayError: Not retryable if the rectangle or slide data is
                invalid or the rectangle misses the slide; retryable if writing
                the mask fails.
        """
        from model_service.providers.empaia.app_service.input_files import (
            write_rectangle_mask,
        )
        from model_service.providers.empaia.app_service.responses import rectangle

        region = rectangle(await gateway.get_input_data(item.name))
        slide = await gateway.slide_info(item.slide or "slide")
        target = directory / MASK_FILE
        await asyncio.to_thread(write_rectangle_mask, target, slide, region)
        return target
