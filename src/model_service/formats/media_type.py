"""Versioned vendor media types: `application/vnd.rationai.<name>+<ext>; version=N`."""

from __future__ import annotations


class UnsupportedFormat(ValueError):
    """A stored output whose media type or version this reader does not know."""


def media_type(name: str, suffix: str, version: int) -> str:
    """Spell a format's media type.

    Args:
        name: The format's name in the type, e.g. `score-grid`.
        suffix: The container, e.g. `npz`.
        version: The layout version.

    Returns:
        E.g. `application/vnd.rationai.score-grid+npz; version=1`.
    """
    return f"application/vnd.rationai.{name}+{suffix}; version={version}"


def require(given: str, known: tuple[str, ...]) -> None:
    """Refuse a media type that is none of the versions a reader knows.

    Args:
        given: The stored output's media type.
        known: The media types the reader can read.

    Raises:
        UnsupportedFormat: If `given` is not one of `known`.
    """
    normalised = ";".join(part.strip() for part in given.split(";"))
    if normalised not in {";".join(k.split("; ")) for k in known}:
        raise UnsupportedFormat(
            f"cannot read `{given}`; this reader knows {', '.join(known)}"
        )
