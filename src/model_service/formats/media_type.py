"""What a data format declares, and its versioned vendor media type.

Each format module declares one `FORMAT`; its media type is
`application/vnd.rationai.<type name>+<container>; version=N`.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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


def normalised(value: str) -> str:
    """A media type with the whitespace around its parameters removed.

    Args:
        value: E.g. `application/x; version=1`.

    Returns:
        E.g. `application/x;version=1`.
    """
    return ";".join(part.strip() for part in value.split(";"))


def require(given: str, known: tuple[str, ...]) -> None:
    """Refuse a media type that is none of the versions a reader knows.

    Args:
        given: The stored output's media type.
        known: The media types the reader can read.

    Raises:
        UnsupportedFormat: If `given` is not one of `known`.
    """
    if normalised(given) not in {normalised(k) for k in known}:
        raise UnsupportedFormat(
            f"cannot read `{given}`; this reader knows {', '.join(known)}"
        )


@dataclass(frozen=True)
class Format:
    """A data format's declaration: what callers learn without opening its module.

    Attributes:
        name: The name sinks declare in `format`, e.g. `score_grid`.
        version: The layout version; a layout change bumps it.
        extension: The file name suffix, without the dot.
        type_name: Its name in the media type, e.g. `score-grid`.
        container: The media type's suffix, e.g. `npz` or `hdf5`.
        module: The module that declares it, filled in by the scan in
            `known.py`; its docstring is the format's reference page.
    """

    name: str
    version: int
    extension: str
    type_name: str
    container: str
    module: str = field(default="", compare=False)

    @property
    def media_type(self) -> str:
        """The versioned vendor media type outputs of this format are stored with."""
        return media_type(self.type_name, self.container, self.version)

    @property
    def known(self) -> tuple[str, ...]:
        """Every media type the format's reader takes: its own version."""
        return (self.media_type,)
