"""A resolved catalog: every document of every kind, checked."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class Entry:
    """One resolved catalog document.

    Attributes:
        id: The document's ID, its file name without `.yaml`.
        file: The base file, relative to the catalog's parent directory.
        document: The document after the environment's patch was merged in.
        value: What the document kind resolved it to.
    """

    id: str
    file: str
    document: dict[str, Any]
    value: Any


@dataclass(frozen=True)
class Catalog:
    """Every document of a catalog directory for one environment.

    Attributes:
        environment: The environment whose patches were applied, or None.
        by_folder: Entries by document folder, then by ID.
    """

    environment: str | None
    by_folder: Mapping[str, Mapping[str, Entry]] = field(default_factory=dict)

    def entries(self, folder: str) -> Mapping[str, Entry]:
        """The entries of one document kind.

        Args:
            folder: The kind's folder, e.g. `served-models`.

        Returns:
            Entries by ID; empty if the folder has none.
        """
        return self.by_folder.get(folder, {})
