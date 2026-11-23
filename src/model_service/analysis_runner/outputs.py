"""All-or-nothing outputs: sinks write into staging directories, committed together.

Each output gets a hidden directory next to its destination, so moving the
artifact into place is a rename on one file system. Commit moves every
artifact only after every sink closed, and puts back what it replaced if one
move fails. Staging directories are removed on exit, success or not.
"""

from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING, Self
from uuid import uuid4

from model_service.contracts.failures import InvalidInvocation
from model_service.directories import make_shared_directory


if TYPE_CHECKING:
    from collections.abc import Collection, Mapping
    from pathlib import Path
    from types import TracebackType


class Staging:
    """Staging directories for one run's outputs."""

    def __init__(self, destinations: Mapping[str, Path]) -> None:
        """Plan a staging directory per output.

        Args:
            destinations: Output name to the path its artifact ends up at.
        """
        self._destinations = dict(destinations)
        self._dirs = {
            name: path.parent / f".{path.name}.{uuid4().hex}.partial"
            for name, path in self._destinations.items()
        }

    def __enter__(self) -> Self:
        """Create the staging directories (and missing destination folders).

        Returns:
            The staging.
        """
        for directory in self._dirs.values():
            # Open to all users, like every directory the API may have to remove.
            make_shared_directory(directory)
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Remove every staging directory and whatever is left in it.

        Args:
            kind: The exception type, if the block raised.
            error: The exception, if the block raised.
            traceback: Its traceback.
        """
        for directory in self._dirs.values():
            shutil.rmtree(directory, ignore_errors=True)

    def work_dir(self, output: str) -> Path:
        """Where the sink of `output` writes.

        Args:
            output: The output name.

        Returns:
            Its staging directory.
        """
        return self._dirs[output]

    def commit(self, artifacts: Mapping[str, Path]) -> dict[str, Path]:
        """Move every artifact to its destination, or none of them.

        Args:
            artifacts: Output name to the file its sink wrote.

        Returns:
            Output name to the destination path.

        Raises:
            RuntimeError: If a sink wrote no regular file in its directory.
            OSError: If a move failed; destinations are as they were.
        """
        for name, artifact in artifacts.items():
            if not artifact.is_file() or artifact.parent != self._dirs[name]:
                raise RuntimeError(
                    f"the sink of {name!r} wrote no file in its work directory"
                )
        moved: list[tuple[Path, Path | None]] = []
        try:
            for name, artifact in artifacts.items():
                destination = self._destinations[name]
                previous = None
                if destination.exists():
                    previous = self._dirs[name] / f".previous.{uuid4().hex}"
                    os.replace(destination, previous)
                moved.append((destination, previous))
                os.replace(artifact, destination)
        except BaseException:
            for destination, previous in reversed(moved):
                if previous is not None and previous.exists():
                    os.replace(previous, destination)
                elif previous is None:
                    destination.unlink(missing_ok=True)
            raise
        return {name: self._destinations[name] for name in artifacts}


def check_destinations(
    destinations: Mapping[str, Path], outputs: Collection[str], inputs: Collection[Path]
) -> None:
    """Refuse destinations that don't match the outputs or would replace an input.

    Args:
        destinations: Output name to path, as the caller gives them.
        outputs: The analysis's output names.
        inputs: The run's input paths.

    Raises:
        InvalidInvocation: For a missing or unknown output, or a destination
            that is one of the inputs.
    """
    missing = sorted(set(outputs) - set(destinations))
    unknown = sorted(set(destinations) - set(outputs))
    if missing or unknown:
        raise InvalidInvocation(
            f"outputs need a path each: {', '.join(sorted(outputs))}; "
            f"missing {missing or 'none'}, unknown {unknown or 'none'}"
        )
    sources = {path.resolve() for path in inputs}
    if any(path.resolve() in sources for path in destinations.values()):
        raise InvalidInvocation("output must not replace an input")
