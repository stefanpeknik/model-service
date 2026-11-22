"""Document kinds: what each catalog folder holds and how it is resolved.

A kind owns one folder (`served-models/`, `analyses/`) and turns each merged
document into its resolved form through a `ResolveContext`, which reports
mistakes in the standard format. Kinds resolve in order, so a later kind can
refer to documents of an earlier one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar


if TYPE_CHECKING:
    from model_service.catalog.context import ResolveContext


class DocumentKind(ABC):
    """How the documents of one catalog folder are checked and resolved.

    Attributes:
        folder: The folder under the catalog root, e.g. `served-models`.
        title: The document kind in errors, e.g. `served model`.
    """

    folder: ClassVar[str]
    title: ClassVar[str]

    @abstractmethod
    def resolve(self, document: dict[str, Any], ctx: ResolveContext) -> Any:
        """Resolve one document, reporting every mistake through `ctx`.

        Args:
            document: The document after the environment's patch.
            ctx: Resolves components and references and collects errors.

        Returns:
            The resolved document; ignored if `ctx` collected errors.
        """
