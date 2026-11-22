"""`value` v1: one scalar result per slide, e.g. a probability or a class name.

Media type `application/vnd.rationai.value+json; version=1`. A JSON object
`{"value": <scalar>}`, where the scalar is a bool, a number or a string. The
EMPAIA `primitive` publisher posts it as the primitive type the analysis's
`expose.empaia` names.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from model_service.formats.media_type import UnsupportedFormat, media_type, require


if TYPE_CHECKING:
    from pathlib import Path


NAME = "value"
VERSION = 1
EXTENSION = "json"
"""The file name suffix, without the dot."""
MEDIA_TYPE = media_type("value", EXTENSION, VERSION)
KNOWN = (MEDIA_TYPE,)

type Scalar = bool | int | float | str

__all__ = [
    "EXTENSION",
    "KNOWN",
    "MEDIA_TYPE",
    "NAME",
    "VERSION",
    "Scalar",
    "UnsupportedFormat",
    "check",
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


def read(path: Path, media_type: str = MEDIA_TYPE) -> Scalar:
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
    require(media_type, KNOWN)
    stored = json.loads(path.read_text(encoding="utf-8"))
    scalar = stored.get("value") if isinstance(stored, dict) else None
    if not isinstance(scalar, bool | int | float | str):
        raise ValueError("`value` is not a bool, number or string")
    return scalar


def check(path: Path) -> list[str]:
    """Check a file against the v1 layout.

    Args:
        path: The file.

    Returns:
        One sentence per problem; empty when the file is a valid value.
    """
    try:
        read(path)
    except (OSError, json.JSONDecodeError) as error:
        return [f"not JSON: {error}"]
    except ValueError as error:
        return [str(error)]
    return []
