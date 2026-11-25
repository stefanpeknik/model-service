"""`collection`: several rectangles, polygons or circles drawn on the slide, as regions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import Field

from components import EmpaiaInput, InputConverter, Options
from components.providers.empaia.converters._drawn import ROI_CLASS


if TYPE_CHECKING:
    from pathlib import Path

    from model_service.providers.empaia.app_service.gateway import EmpaiaGateway


class CollectionOptions(Options):
    """Options of `collection`."""

    items: Literal["rectangle", "polygon", "circle"] = Field(
        description="The shape users draw, as many times as they like."
    )


class Collection(InputConverter):
    """Any number of one drawn shape on the app's slide, given to the pipeline as regions.

    It becomes a `regions` file with one region per item, in the
    collection's order, each keeping its annotation's ID; a circle becomes a
    64-sided polygon. Each item carries the region-of-interest class.
    """

    name = "collection"
    ead_type = "collection"
    Options = CollectionOptions
    example_options: ClassVar[dict[str, Any]] = {"items": "polygon"}
    options: CollectionOptions

    def io_fragment(self, item: EmpaiaInput) -> dict[str, Any]:
        """Declare a collection of region-of-interest shapes on the app's slide.

        Args:
            item: The input; `slide` names the app's slide.

        Returns:
            The collection entry, its items referencing the slide.
        """
        return {
            "type": self.ead_type,
            "items": {
                "type": self.options.items,
                "reference": f"io.{item.slide}",
                "classes": [ROI_CLASS],
            },
        }

    async def materialize(
        self, gateway: EmpaiaGateway, item: EmpaiaInput, directory: Path
    ) -> Path:
        """Read the collection and write its items as regions into `directory`.

        Args:
            gateway: The job's gateway.
            item: The input.
            directory: Where to write `regions.geojson`.

        Returns:
            The regions file.

        Raises:
            EmpaiaGatewayError: Not retryable if an item is invalid; retryable
                if writing fails.
        """
        from model_service.providers.empaia.app_service.regions import (
            materialize_collection,
        )

        return await materialize_collection(
            gateway, item.name, self.options.items, directory
        )
