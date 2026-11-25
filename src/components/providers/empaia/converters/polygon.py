"""`polygon`: a polygon drawn on the slide, as a region in the `regions` format."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from components import EmpaiaInput, InputConverter
from components.providers.empaia.converters.rectangle import ROI_CLASS


if TYPE_CHECKING:
    from pathlib import Path

    from model_service.providers.empaia.app_service.gateway import EmpaiaGateway


class Polygon(InputConverter):
    """A polygon on the app's slide, given to the pipeline as one region.

    It becomes a `regions` file with one polygon in level-0 pixels, keeping
    the annotation's ID. EMPAIA offers it with the region-of-interest class,
    so the viewer lets a user draw it.
    """

    name = "polygon"
    ead_type = "polygon"

    def io_fragment(self, item: EmpaiaInput) -> dict[str, Any]:
        """Declare a region-of-interest polygon on the app's slide.

        Args:
            item: The input; `slide` names the app's slide.

        Returns:
            The polygon entry, referencing the slide.
        """
        return {
            "type": self.ead_type,
            "reference": f"io.{item.slide}",
            "classes": [ROI_CLASS],
        }

    async def materialize(
        self, gateway: EmpaiaGateway, item: EmpaiaInput, directory: Path
    ) -> Path:
        """Read the polygon and write it as a region into `directory`.

        Args:
            gateway: The job's gateway.
            item: The input.
            directory: Where to write `regions.geojson`.

        Returns:
            The regions file.

        Raises:
            EmpaiaGatewayError: Not retryable if the polygon is invalid;
                retryable if writing fails.
        """
        from model_service.providers.empaia.app_service.regions import (
            materialize_regions,
        )

        return await materialize_regions(gateway, item.name, self.ead_type, directory)
