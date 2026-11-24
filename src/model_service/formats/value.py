"""`value` v1: one scalar result per slide, e.g. a probability or a class name.

Media type `application/vnd.rationai.value+json; version=1`. A JSON object
`{"value": <scalar>}`, where the scalar is a bool, a number or a string. The
EMPAIA `primitive` publisher posts it as the primitive type the pipeline's
`providers.empaia` names.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from model_service.formats.media_type import Format, UnsupportedFormat, require


if TYPE_CHECKING:
    from pathlib import Path


FORMAT = Format(
    name="value", version=1, extension="json", type_name="value", container="json"
)

type Scalar = bool | int | float | str

__all__ = [
    "FORMAT",
    "Scalar",
    "UnsupportedFormat",
    "read",
    "write",
]


def write(path: Path, scalar: Scalar) -> None:
    """Write one scalar.

    Args:
        path: File to write.
        scalar: The value.
    """
    path.write_text(json.dumps({"value": scalar}, allow_nan=False), encoding="utf-8")


def read(path: Path, media_type: str = FORMAT.media_type) -> Scalar:
    """Read the scalar.

    Args:
        path: The file.
        media_type: The media type the file was stored with.

    Returns:
        The value.

    Raises:
        UnsupportedFormat: For a media type or version this reader doesn't know.
        ValueError: If the file is not JSON or holds no scalar.
    """
    require(media_type, FORMAT.known)
    stored = json.loads(path.read_text(encoding="utf-8"))
    scalar = stored.get("value") if isinstance(stored, dict) else None
    if not isinstance(scalar, bool | int | float | str):
        raise ValueError("`value` is not a bool, number or string")
    return scalar
