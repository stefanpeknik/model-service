"""The data formats this repo knows, by the name sinks declare in `format`.

Every module of this package that defines `NAME` and `MEDIA_TYPE` is a
format; nothing lists them by hand.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass

import model_service.formats


@dataclass(frozen=True)
class Format:
    """What callers learn about a format without opening its module.

    Attributes:
        name: The name sinks declare, e.g. `score_grid`.
        media_type: The versioned vendor media type.
        extension: The file name suffix, without the dot.
        known: Every media type its reader takes, older layouts included.
        module: The module that declares it; its docstring is its reference page.
    """

    name: str
    media_type: str
    extension: str
    known: tuple[str, ...] = ()
    module: str = ""


def _scan() -> dict[str, Format]:
    """Import every format module of this package, by format name."""
    found: dict[str, Format] = {}
    package = model_service.formats
    for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda m: m.name):
        module = importlib.import_module(f"{package.__name__}.{info.name}")
        if not hasattr(module, "NAME") or not hasattr(module, "MEDIA_TYPE"):
            continue
        found[module.NAME] = Format(
            module.NAME,
            module.MEDIA_TYPE,
            module.EXTENSION,
            module.KNOWN,
            module.__name__,
        )
    return found


FORMATS = _scan()


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
    """A media type with the whitespace around its parameters removed.

    Args:
        media_type: E.g. `application/x; version=1`.

    Returns:
        E.g. `application/x;version=1`.
    """
    return ";".join(part.strip() for part in media_type.split(";"))
