"""The data formats this repo knows, by the name sinks declare in `format`."""

from __future__ import annotations

from dataclasses import dataclass

from model_service.formats import embedding_table, score_grid, value


@dataclass(frozen=True)
class Format:
    """What callers learn about a format without opening its module.

    Attributes:
        name: The name sinks declare, e.g. `score_grid`.
        media_type: The versioned vendor media type.
        extension: The file name suffix, without the dot.
        known: Every media type its reader takes, older layouts included.
    """

    name: str
    media_type: str
    extension: str
    known: tuple[str, ...] = ()


FORMATS = {
    f.name: f
    for f in (
        Format(
            score_grid.NAME,
            score_grid.MEDIA_TYPE,
            score_grid.EXTENSION,
            score_grid.KNOWN,
        ),
        Format(
            embedding_table.NAME,
            embedding_table.MEDIA_TYPE,
            embedding_table.EXTENSION,
            embedding_table.KNOWN,
        ),
        Format(value.NAME, value.MEDIA_TYPE, value.EXTENSION, value.KNOWN),
    )
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


def format_of(media_type: str) -> Format | None:
    """The format a stored output's media type names.

    Args:
        media_type: The stored output's media type; spacing around `;` is ignored.

    Returns:
        The format whose reader takes it, or None.
    """
    given = _normalised(media_type)
    return next(
        (f for f in FORMATS.values() if given in {_normalised(k) for k in f.known}),
        None,
    )


def _normalised(media_type: str) -> str:
    return ";".join(part.strip() for part in media_type.split(";"))
