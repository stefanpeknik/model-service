"""`region_values` v1: one number per region, e.g. the share of tumour in each drawn area.

Media type `application/vnd.rationai.region-values+json; version=1`. A JSON
object `{"regions": [{"id": <string>, "value": <number or null>}, ...]}`,
one entry per region of the run's `regions` input, in its order; `id` is the
region's. `value` is null where the run had nothing to measure, e.g. a region
over glass. A run without regions has one entry, `slide`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from model_service.formats.media_type import Format, UnsupportedFormat, require


if TYPE_CHECKING:
    from pathlib import Path


FORMAT = Format(
    name="region_values",
    version=1,
    extension="json",
    type_name="region-values",
    container="json",
)

__all__ = [
    "FORMAT",
    "RegionValue",
    "UnsupportedFormat",
    "read",
    "write",
]


@dataclass(frozen=True)
class RegionValue:
    """One region's number.

    Attributes:
        id: The region's ID.
        value: The number, or None where there was nothing to measure.
    """

    id: str
    value: float | None


def write(path: Path, values: tuple[RegionValue, ...]) -> None:
    """Write the regions' values.

    Args:
        path: File to write.
        values: One value per region, in region order.
    """
    path.write_text(
        json.dumps(
            {"regions": [{"id": v.id, "value": v.value} for v in values]},
            allow_nan=False,
        ),
        encoding="utf-8",
    )


def read(path: Path, media_type: str = FORMAT.media_type) -> tuple[RegionValue, ...]:
    """Read the regions' values.

    Args:
        path: The file.
        media_type: The media type the file was stored with.

    Returns:
        The values, in region order.

    Raises:
        UnsupportedFormat: For a media type or version this reader doesn't know.
        ValueError: If the file is not such an object.
    """
    require(media_type, FORMAT.known)
    stored = json.loads(path.read_text(encoding="utf-8"))
    entries = stored.get("regions") if isinstance(stored, dict) else None
    if not isinstance(entries, list):
        raise ValueError("`regions` is not a list")
    values: list[RegionValue] = []
    for entry in entries:
        region_id = entry.get("id") if isinstance(entry, dict) else None
        value = entry.get("value") if isinstance(entry, dict) else None
        if not isinstance(region_id, str):
            raise ValueError("every entry needs a string `id`")
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError(f"the value of {region_id!r} is not a finite number")
        values.append(RegionValue(region_id, None if value is None else float(value)))
    return tuple(values)
