"""What a builder author implements and receives: `Builder` and `BuilderContext`.

See `docs/guides/adding-builders.md`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from model_service.builders.artifacts import atomic_artifact
from model_service.builders.contracts import (
    BuilderInputs,
    BuilderOutputs,
    SharedArtifact,
)
from model_service.builders.model_clients import ModelClient


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from datetime import datetime
    from pathlib import Path

    from pydantic import BaseModel

    from model_service.builders.access import InputAccess


@dataclass(frozen=True, slots=True)
class BuilderContext[K: ModelClient[Any], I: BuilderInputs, C: BaseModel]:
    """What one builder run gets: its model, inputs and configuration, typed.

    The application checks the model against the builder's contract and its
    running deployment against its declaration, resolves the inputs to local files
    and the configuration with the builder's own classes before building it, so
    a builder can rely on these types without checking them again.
    """

    access: InputAccess
    # The verified client of the model's contract; `model.declaration` is what
    # the caller declared for it.
    model: K
    inputs: I
    configuration: C
    deadline_at: datetime
    output_paths: Mapping[str, str]

    def output_path(self, output: str) -> Path:
        """Return where to write a declared artifact output.

        Uses the invocation's `output_path` override when it set one, otherwise the
        output's default path. `write_artifact` is the usual way to write one.

        Args:
            output: Output name from the builder's declaration.

        Returns:
            Local path for the output file.

        Raises:
            KeyError: The output has no path, i.e. it is not an artifact output.
        """
        return self.access.output_path(self.output_paths[output])

    @contextmanager
    def write_artifact(self, output: str) -> Iterator[PendingArtifact]:
        """Write an artifact output atomically.

        The builder writes `pending.path`, a temporary file next to the
        destination; on success it replaces the destination, and on failure it
        is removed, so a failed run leaves no partial artifact. After the block,
        `pending.artifact` is the output to return.

        Args:
            output: Output name from the builder's declaration.

        Yields:
            The pending artifact.

        Raises:
            InvalidInvocation: The destination is one of the inputs.
            RuntimeError: The block did not write a regular file.
        """
        destination = self.output_path(output)
        with atomic_artifact(destination, inputs=self.inputs.paths()) as temporary:
            yield PendingArtifact(path=temporary, destination=destination)


@dataclass(frozen=True, slots=True)
class PendingArtifact:
    """An artifact being written by `BuilderContext.write_artifact`."""

    # Where the builder writes; closed before the block ends.
    path: Path
    destination: Path

    @property
    def artifact(self) -> SharedArtifact:
        """The finished artifact, to return in the builder's outputs."""
        return SharedArtifact(path=str(self.destination))


class Builder[
    K: ModelClient[Any],
    I: BuilderInputs,
    C: BaseModel,
    R: BuilderOutputs,
](ABC):
    """Produces a builder's outputs from a slide and a model.

    `K` is the model client of the contract it accepts; `I`, `C` and `R` are its
    declared inputs, configuration and outputs models.
    """

    @abstractmethod
    async def run(self, context: BuilderContext[K, I, C]) -> R:
        """Produce the builder's outputs for one invocation.

        Args:
            context: Verified model client, resolved inputs, configuration and
                deadline.

        Returns:
            The builder's outputs model, one field per declared output.
        """
        ...
