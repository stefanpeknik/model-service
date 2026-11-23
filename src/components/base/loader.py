"""Loader: fetches weights to a local path."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from components.base.component import Component
from components.base.slots import STEP_WEIGHTS, WEIGHTS


if TYPE_CHECKING:
    from pathlib import Path

    from components.base.slots import Slot


class Loader(Component):
    """Fetches a model's weights (an MLflow artifact, a Hugging Face repo, ...).

    Used under a served model's `weights:` and under the `weights:` of an
    analysis step that carries its own small model.
    """

    kind: ClassVar[str] = "loader"
    slots: ClassVar[tuple[Slot, ...]] = (WEIGHTS, STEP_WEIGHTS)

    @abstractmethod
    def fetch(self, cache_dir: Path) -> Path:
        """Make the weights available locally.

        Args:
            cache_dir: A directory the loader may download into and reuse.

        Returns:
            The local path of the weights (a file or a directory).
        """

    @abstractmethod
    def revision(self) -> str:
        """Name the exact weights this loader fetches, for `describe()`.

        After `fetch` it may also name what was fetched, e.g. a checksum of
        the downloaded file.

        Returns:
            A pinned reference: an artifact URI, a repo revision, a path.
        """

    @classmethod
    def offline_options(cls, weights: Path) -> dict[str, Any] | None:
        """Options that fetch `weights` without a network, for the conformance suite.

        Args:
            weights: A stand-in weights file on this machine.

        Returns:
            The options, or None if the loader only fetches over a network;
            the suite then checks `revision()` but not `fetch`.
        """
        return None

    def expected_revision(self) -> str | None:
        """The revision `revision()` must give after `fetch`, if the options pin it.

        The analysis runner compares it with the served model's `describe()`,
        so weights that changed under the same options are refused.

        Returns:
            The pinned revision, or None when the options don't pin the content.
        """
        return None
