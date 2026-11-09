"""How builders reach their inputs: the interface providers implement.

A provider's access adapter turns an invocation's provider-specific inputs into
local files for one run, and says where outputs go; builders see only the
resulting paths.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Awaitable
    from pathlib import Path

    from model_service.builders.contracts import (
        ExecutionContext,
        Invocation,
        ModelBinding,
    )


class InputAccess(ABC):
    @abstractmethod
    def input_path(self, name: str) -> Path:
        """Resolve a named invocation input to a local file.

        Args:
            name: Input name from the builder's declaration.

        Returns:
            Local path the builder can read the input from.
        """
        ...

    @abstractmethod
    def output_path(self, relative_path: str) -> Path:
        """Resolve an output's relative path to the local file to write.

        Args:
            relative_path: Output path relative to the invocation's output location.

        Returns:
            Local path the builder writes the artifact to.
        """
        ...

    @abstractmethod
    def is_visible_artifact(self, path: Path) -> bool:
        """Check that a returned artifact is a file this access exposes to callers.

        Args:
            path: Artifact path a builder returned in its result.

        Returns:
            True when the path is a file inside the access's visible output area.
        """
        ...

    @abstractmethod
    async def aclose(self) -> None:
        """Release what the access holds, such as fetched inputs or temporary files."""
        ...


class AccessAdapter(ABC):
    # The model tells an adapter that fetches slides which resolution the builder reads.
    @abstractmethod
    def create(
        self,
        invocation: Invocation,
        execution_context: ExecutionContext,
        model: ModelBinding,
    ) -> InputAccess | Awaitable[InputAccess]:
        """Create the input access for one invocation attempt.

        Args:
            invocation: Invocation whose inputs and outputs the access resolves.
            execution_context: Attempt identity and protected provider state.
            model: Model binding; tells an adapter that fetches slides which
                resolution the builder reads.

        Returns:
            The access, or an awaitable resolving to it.
        """
        ...
