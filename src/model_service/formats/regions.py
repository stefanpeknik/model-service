"""`regions` v1: named areas on a slide, such as regions of interest a user drew.

Media type `application/vnd.rationai.regions+json; version=1`. A GeoJSON
`FeatureCollection` whose coordinates are pixels of the slide's level 0,
x to the right and y down. Each feature is one region: its `id` is a
string, unique in the file, and its geometry a `Polygon` or `MultiPolygon`.
An empty collection selects nothing. A pipeline takes it as an input, e.g.
`region_of_interest`; the EMPAIA rectangle, polygon, circle and collection
converters write it from the annotations a user drew.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from model_service.formats.media_type import Format, UnsupportedFormat, require


if TYPE_CHECKING:
    from pathlib import Path

    from shapely.geometry.base import BaseGeometry


FORMAT = Format(
    name="regions",
    version=1,
    extension="geojson",
    type_name="regions",
    container="json",
)

AREAS = frozenset({"Polygon", "MultiPolygon"})

__all__ = [
    "FORMAT",
    "Region",
    "UnsupportedFormat",
    "read",
    "write",
]


@dataclass(frozen=True)
class Region:
    """One named area.

    Attributes:
        id: Unique in its file; an EMPAIA region keeps its annotation's ID.
        geometry: A polygon or multipolygon in level-0 pixels.
    """

    id: str
    geometry: BaseGeometry


def write(path: Path, regions: tuple[Region, ...]) -> None:
    """Write regions as a GeoJSON feature collection.

    Args:
        path: File to write.
        regions: The regions, in order.
    """
    from shapely.geometry import mapping

    path.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "id": region.id,
                        "properties": {},
                        "geometry": mapping(region.geometry),
                    }
                    for region in regions
                ],
            }
        ),
        encoding="utf-8",
    )


def read(path: Path, media_type: str = FORMAT.media_type) -> tuple[Region, ...]:
    """Read the regions, each geometry made valid.

    Args:
        path: The file.
        media_type: The media type the file was stored with.

    Returns:
        The regions, in file order.

    Raises:
        UnsupportedFormat: For a media type or version this reader doesn't know.
        ValueError: If the file is not a feature collection of polygons with
            unique string IDs.
    """
    from shapely import make_valid
    from shapely.geometry import shape

    require(media_type, FORMAT.known)
    stored = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(stored, dict) or stored.get("type") != "FeatureCollection":
        raise ValueError("regions must be a GeoJSON FeatureCollection")
    regions: list[Region] = []
    for feature in stored.get("features") or []:
        region_id = feature.get("id") if isinstance(feature, dict) else None
        geometry = feature.get("geometry") if isinstance(feature, dict) else None
        if not isinstance(region_id, str) or not region_id:
            raise ValueError("every region needs a string `id`")
        if not isinstance(geometry, dict) or geometry.get("type") not in AREAS:
            raise ValueError(f"region {region_id!r} is not a Polygon or MultiPolygon")
        regions.append(Region(region_id, make_valid(shape(geometry))))
    if len({r.id for r in regions}) != len(regions):
        raise ValueError("region IDs must be unique")
    return tuple(regions)
