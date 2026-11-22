"""The model catalog mounted from the chart's ConfigMap, reloaded in place.

The ConfigMap is mounted as a directory without `subPath`, so kubelet updates
its files when a Helm upgrade changes the catalog, without restarting the
API. `MountedCatalog` notices the change, reads the new files (see
`catalog_files.py`) and swaps them in only if they load. On failure it keeps
the catalog it has, logs why, and reports it under `/healthz/catalog` until a
later version loads. Executions carry the analysis document resolved at
admission, so a swap does not touch them.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from model_service.api.catalog_files import CatalogError, read_catalog
from model_service.catalog.analyses import FOLDER as ANALYSES
from model_service.catalog.served_models import FOLDER as SERVED_MODELS


if TYPE_CHECKING:
    from pathlib import Path

    from model_service.catalog import Catalog


logger = logging.getLogger(__name__)


class CatalogState(StrEnum):
    """Whether the mounted files are the catalog in use."""

    LOADED = "loaded"
    RELOAD_FAILED = "reload_failed"


@dataclass(frozen=True, slots=True)
class CatalogStatus:
    """The catalog's state, as `/healthz/catalog` reports it.

    Attributes:
        status: `loaded`, or `reload_failed` while the API keeps an older catalog.
        error: Why the mounted files did not load, in the catalog's error format.
    """

    status: CatalogState
    error: str | None = None


class MountedCatalog:
    """The catalog from a mounted directory, swapped when its files change."""

    def __init__(self, directory: Path) -> None:
        """Load the mounted catalog.

        Args:
            directory: Where the ConfigMap is mounted.

        Raises:
            CatalogError: If the catalog does not load; with no older catalog
                to keep, the API cannot start.
        """
        self._directory = directory
        self._fingerprint = _fingerprint(directory)
        self._current: Catalog = read_catalog(directory)
        self._error: str | None = None

    def refresh(self) -> bool:
        """Load the mounted files if they changed since the last look.

        Returns:
            True if a new catalog replaced the old one.
        """
        seen = _fingerprint(self._directory)
        if seen == self._fingerprint:
            return False
        try:
            catalog = read_catalog(self._directory)
        except CatalogError as error:
            # Remembered, so the same broken files are reported once.
            self._fingerprint = seen
            self._error = str(error)
            logger.error(
                "model catalog not reloaded, the previous one stays: %s", error
            )
            return False
        if _fingerprint(self._directory) != seen:
            # kubelet swapped the files mid-read; the next look reads them again.
            return False
        self._current, self._fingerprint, self._error = catalog, seen, None
        logger.info(
            "model catalog reloaded: %s; analyses: %s",
            ", ".join(catalog.entries(SERVED_MODELS)) or "no models",
            ", ".join(catalog.entries(ANALYSES)) or "none",
        )
        return True

    async def watch(self, interval: float) -> None:
        """Refresh every `interval` seconds, until cancelled.

        Args:
            interval: Seconds between looks.
        """
        while True:
            await asyncio.sleep(interval)
            await asyncio.to_thread(self.refresh)

    def status(self) -> CatalogStatus:
        """Report whether the mounted files are the catalog in use.

        Returns:
            The state and, after a failed reload, why it failed.
        """
        if self._error is None:
            return CatalogStatus(CatalogState.LOADED)
        return CatalogStatus(CatalogState.RELOAD_FAILED, self._error)

    @property
    def resolved(self) -> Catalog:
        """The served models and analyses in use, as the catalog loader resolved them."""
        return self._current


def _fingerprint(directory: Path) -> str:
    """Hash the mounted files' names and contents.

    kubelet's own entries (`..data` and the timestamped folders) are skipped;
    the files are symlinks into them, followed.
    """
    digest = hashlib.sha256()
    try:
        for path in sorted(directory.iterdir()):
            if path.name.startswith(".") or not path.is_file():
                continue
            digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    except OSError:
        return "unreadable"
    return digest.hexdigest()
