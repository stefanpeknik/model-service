"""The data formats this repo knows, by the name sinks declare in `format`."""

from __future__ import annotations

from dataclasses import dataclass

from model_service.formats import score_grid


@dataclass(frozen=True)
class Format:
    """What callers learn about a format without opening its module.

    Attributes:
        name: The name sinks declare, e.g. `score_grid`.
        media_type: The versioned vendor media type.
        extension: The file name suffix, without the dot.
    """

    name: str
    media_type: str
    extension: str


FORMATS = {
    f.name: f
    for f in (Format(score_grid.NAME, score_grid.MEDIA_TYPE, score_grid.EXTENSION),)
}


def get_format(name: str) -> Format:
    """Look up a format by name.

    Args:
        name: The format's name.

    Returns:
        The format.

    Raises:
        KeyError: If no format module declares that name.
    """
    return FORMATS[name]
