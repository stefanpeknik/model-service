"""What a builder author implements and receives: `Builder`, `BuilderContext`, settings."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from model_service.builders.declarations import BuilderInputs, BuilderOutputs
from model_service.contracts.failures import InvalidInvocation
from model_service.contracts.result import SharedArtifact
from model_service.deployed_models.model_clients import ModelClient


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from model_service.builders.input_access import InputAccess


class BuilderSettings(BaseModel):
    """Per-replica limits a builder reads, from its Ray `user_config`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    num_threads: int = Field(default=2, gt=0)
    max_concurrent_tasks: int = Field(default=4, gt=0)


@dataclass(frozen=True, slots=True)
class BuilderContext[K: ModelClient[Any], I: BuilderInputs, C: BaseModel]:
    """What one builder run gets: its model, inputs and configuration, typed.

    The model is checked against the builder's contract and its running
    deployment against its declaration, and the inputs are resolved to local
    files and the configuration with the builder's own classes before the
    context is built, so a builder can rely on these types without checking
    them again.
    """

    access: InputAccess
    # The verified client of the model's contract; `model.declaration` is the
    # model's declaration.
    model: K
    inputs: I
    configuration: C
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
    def write_artifact(self, output: str) -> Iterator[tuple[Path, SharedArtifact]]:
        """Write an artifact output atomically.

        The builder writes the yielded path, a temporary file next to the
        destination; on success it replaces the destination, and on failure it
        is removed, so a failed run leaves no partial artifact. The yielded
        artifact is the output to return.

        Args:
            output: Output name from the builder's declaration.

        Yields:
            The temporary path, and the finished artifact.

        Raises:
            InvalidInvocation: The destination is one of the inputs.
            RuntimeError: The block did not write a regular file.
        """
        destination = self.output_path(output)
        with _atomic_artifact(destination, inputs=self.inputs.paths()) as temporary:
            yield temporary, SharedArtifact(path=str(destination))


@contextmanager
def _atomic_artifact(destination: Path, *, inputs: tuple[Path, ...]) -> Iterator[Path]:
    """Yield a sibling temporary path and replace the destination only on success.

    Args:
        destination: Final output path resolved by invocation-scoped storage access.
        inputs: Resolved input paths which publication must not replace.

    Yields:
        Path: Unique temporary path; the caller must close its writer before exit.

    Raises:
        InvalidInvocation: The destination aliases an input path.
        RuntimeError: Successful computation did not produce a regular file.
        OSError: Creating directories, publishing, or removing the temporary file fails.
    """
    if any(destination.resolve() == source.resolve() for source in inputs):
        raise InvalidInvocation("output must not replace an input")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{uuid4().hex}{destination.suffix}"
    )
    try:
        yield temporary
        if not temporary.is_file():
            raise RuntimeError("builder did not produce a regular artifact file")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


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

    def __init__(self, settings: BuilderSettings) -> None:
        self._settings = settings

    @abstractmethod
    async def run(self, context: BuilderContext[K, I, C]) -> R:
        """Produce the builder's outputs for one invocation.

        Args:
            context: Verified model client, resolved inputs and configuration.

        Returns:
            The builder's outputs model, one field per declared output.
        """
        ...
