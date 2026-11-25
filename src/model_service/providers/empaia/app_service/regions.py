"""EMPAIA annotations a user drew, as regions in the `regions` format.

Rectangles, polygons and circles are read with the App Service's field
names, in slide level-0 pixels; shapely builds the geometry, and a circle
becomes a polygon of 64 sides. Each region keeps its annotation's ID; data
without one (a test double) is named after the input.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from pydantic import Field

from model_service.directories import make_shared_directory
from model_service.formats import regions as regions_format
from model_service.providers.empaia.app_service.gateway import EmpaiaGatewayError
from model_service.providers.empaia.app_service.responses import (
    EmpaiaResponse,
    FiniteNumber,
    PositiveNumber,
)


if TYPE_CHECKING:
    from pathlib import Path

    from shapely.geometry.base import BaseGeometry

    from model_service.formats.regions import Region
    from model_service.providers.empaia.app_service.gateway import EmpaiaGateway


REGION_TYPES = ("rectangle", "polygon", "circle")
"""The EAD annotation types that enclose an area."""

type Point = Annotated[list[FiniteNumber], Field(min_length=2, max_length=2)]


class _Annotation(EmpaiaResponse):
    """An annotation's ID, when the App Service gives one."""

    id: str | None = None


class _Rectangle(_Annotation):
    """A rectangle from its upper-left corner and size."""

    upper_left: Point
    width: PositiveNumber
    height: PositiveNumber

    def geometry(self) -> BaseGeometry:
        """The rectangle as a polygon."""
        from shapely import box

        x, y = self.upper_left
        return box(x, y, x + self.width, y + self.height)


class _Polygon(_Annotation):
    """A polygon from its corners."""

    coordinates: Annotated[list[Point], Field(min_length=3)]

    def geometry(self) -> BaseGeometry:
        """The polygon, made valid if it crosses itself."""
        from shapely import Polygon, make_valid

        return make_valid(Polygon(self.coordinates))


class _Circle(_Annotation):
    """A circle from its centre and radius."""

    center: Point
    radius: PositiveNumber

    def geometry(self) -> BaseGeometry:
        """The circle as a 64-sided polygon."""
        from shapely import Point as ShapelyPoint

        return ShapelyPoint(self.center).buffer(self.radius, quad_segs=16)


_MODELS: dict[str, type[_Rectangle | _Polygon | _Circle]] = {
    "rectangle": _Rectangle,
    "polygon": _Polygon,
    "circle": _Circle,
}


def region(value: object, ead_type: str, fallback_id: str) -> Region:
    """Read one drawn annotation as a region.

    Args:
        value: The annotation, as the App Service answers it.
        ead_type: `rectangle`, `polygon` or `circle`.
        fallback_id: The region's ID if the data has none.

    Returns:
        The region, in level-0 pixels.

    Raises:
        EmpaiaGatewayError: Not retryable if the data is not that annotation
            or encloses no area.
    """
    parsed = _MODELS[ead_type].parse(value)
    shape = parsed.geometry()
    if shape.is_empty or shape.area <= 0:
        raise EmpaiaGatewayError.invalid(f"EMPAIA {ead_type} input encloses no area")
    return regions_format.Region(parsed.id or fallback_id, shape)


def collection_regions(value: object, item_type: str, name: str) -> tuple[Region, ...]:
    """Read a collection of drawn annotations as regions.

    Args:
        value: The collection, as the App Service answers it, with its `items`.
        item_type: The items' EAD type.
        name: The input's name, for the IDs of items without one.

    Returns:
        One region per item, in the collection's order.

    Raises:
        EmpaiaGatewayError: Not retryable if the data is not such a collection.
    """
    items = value.get("items") if isinstance(value, dict) else None
    if not isinstance(items, list):
        raise EmpaiaGatewayError.invalid(
            "EMPAIA AppService returned an invalid response"
        )
    return tuple(
        region(item, item_type, f"{name}-{index}") for index, item in enumerate(items)
    )


def write_regions(destination: Path, regions: tuple[Region, ...]) -> None:
    """Write regions in the `regions` format.

    Args:
        destination: The file to write.
        regions: The regions.

    Raises:
        EmpaiaGatewayError: Retryable if the file cannot be written.
    """
    try:
        make_shared_directory(destination.parent)
        regions_format.write(destination, regions)
    except OSError as error:
        raise EmpaiaGatewayError(
            "EMPAIA region input could not be materialized", retryable=True
        ) from error


REGIONS_FILE = "regions.geojson"


async def materialize_regions(
    gateway: EmpaiaGateway,
    name: str,
    ead_type: str,
    directory: Path,
    item_type: str | None = None,
) -> Path:
    """Read a drawn annotation or a collection of them and write its regions.

    Args:
        gateway: The job's gateway.
        name: The input's name.
        ead_type: `rectangle`, `polygon`, `circle` or `collection`.
        directory: Where to write `regions.geojson`.
        item_type: A collection's item type, one of the other three.

    Returns:
        The regions file.

    Raises:
        EmpaiaGatewayError: Not retryable if the data is invalid; retryable if
            writing fails.
    """
    import asyncio

    data = await gateway.get_input_data(name)
    if ead_type in REGION_TYPES:
        found: tuple[Region, ...] = (region(data, ead_type, name),)
    else:
        found = collection_regions(data, item_type or "", name)
    target = directory / REGIONS_FILE
    await asyncio.to_thread(write_regions, target, found)
    return target
