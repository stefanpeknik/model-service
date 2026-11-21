"""The one error format every catalog check reports in."""

from __future__ import annotations

from dataclasses import dataclass


type YamlPath = tuple[str | int, ...]


def render_path(path: YamlPath) -> str:
    """Render a YAML path as `runtime.options` or `post[0].registers`.

    Args:
        path: Mapping keys and list indices from the document root.

    Returns:
        The dotted path, empty for the document itself.
    """
    text = ""
    for part in path:
        text += f"[{part}]" if isinstance(part, int) else f".{part}"
    return text.removeprefix(".")


@dataclass(frozen=True)
class CatalogError:
    """One mistake in a catalog file.

    Renders as `<file>:<yaml path>: <what's wrong>. <what would fit>. Hint:
    <how to fix>`, or without the path when the whole file is meant.

    Attributes:
        file: The file, relative to the catalog's parent directory.
        path: Where in the file, e.g. `runtime.threads`.
        what: What's wrong, without a final period.
        fits: What would fit instead, without a final period.
        hint: How to fix it, without a final period.
    """

    file: str
    path: str
    what: str
    fits: str
    hint: str

    def __str__(self) -> str:
        """Render in the standard format."""
        where = f"{self.file}:{self.path}" if self.path else self.file
        return f"{where}: {self.what}. {self.fits}. Hint: {self.hint}"


class CatalogInvalid(Exception):
    """A catalog that failed to load, with every error found."""

    def __init__(self, errors: list[CatalogError]) -> None:
        """Hold the errors; the message lists them one per line.

        Args:
            errors: Every error, in file order.
        """
        super().__init__("\n".join(map(str, errors)))
        self.errors = errors
