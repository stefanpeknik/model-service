"""The data formats this repo knows, by the name sinks declare in `format`.

Every module of this package that declares a `FORMAT` is a format; nothing
lists them by hand.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import replace

import model_service.formats
from model_service.formats.media_type import Format


def _scan() -> dict[str, Format]:
    """Import every format module of this package, by format name."""
    found: dict[str, Format] = {}
    package = model_service.formats
    for info in sorted(pkgutil.iter_modules(package.__path__), key=lambda m: m.name):
        module = importlib.import_module(f"{package.__name__}.{info.name}")
        declared = getattr(module, "FORMAT", None)
        if isinstance(declared, Format):
            found[declared.name] = replace(declared, module=module.__name__)
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
