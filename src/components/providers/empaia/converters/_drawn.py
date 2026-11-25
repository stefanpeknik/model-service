"""What the converters of shapes a user draws share: one region on the app's slide."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from components import EmpaiaInput, InputConverter


if TYPE_CHECKING:
    from pathlib import Path

    from model_service.providers.empaia.app_service.gateway import EmpaiaGateway


ROI_CLASS = "org.empaia.global.v1.classes.roi"
"""The class EMPAIA offers drawn inputs with, so the viewer lets a user draw them."""


class DrawnShape(InputConverter):
    """A shape drawn on the app's slide, given to the pipeline as one region.

    It becomes a `regions` file with the shape in level-0 pixels, keeping the
    annotation's ID. A subclass names the EAD type.
    """

    def io_fragment(self, item: EmpaiaInput) -> dict[str, Any]:
        """Declare a region-of-interest shape on the app's slide.

        Args:
            item: The input; `slide` names the app's slide.

        Returns:
            The entry, referencing the slide.
        """
        return {
            "type": self.ead_type,
            "reference": f"io.{item.slide}",
            "classes": [ROI_CLASS],
        }

    async def materialize(
        self, gateway: EmpaiaGateway, item: EmpaiaInput, directory: Path
    ) -> Path:
        """Read the shape and write it as a region into `directory`.

        Args:
            gateway: The job's gateway.
            item: The input.
            directory: Where to write `regions.geojson`.

        Returns:
            The regions file.

        Raises:
            EmpaiaGatewayError: Not retryable if the shape is invalid;
                retryable if writing fails.
        """
        from model_service.providers.empaia.app_service.regions import (
            materialize_shape,
        )

        return await materialize_shape(gateway, item.name, self.ead_type, directory)
