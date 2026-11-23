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
            name: path.parent / _staging_name(path)
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
                if destination.is_dir():
                    raise IsADirectoryError(f"{destination} is a directory")
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


# Most file systems cap a file name at 255 bytes.
_NAME_MAX = 255


def _staging_name(destination: Path) -> str:
    """A fresh name for the hidden staging directory of `destination`.

    Args:
        destination: The output's destination path.

    Returns:
        The directory name, unique per call.
    """
    return f".{destination.name}.{uuid4().hex}.partial"


def check_destinations(
    destinations: Mapping[str, Path], outputs: Collection[str], inputs: Collection[Path]
) -> None:
    """Refuse destinations that don't match the outputs or would replace an input.

    Args:
        destinations: Output name to path, as the caller gives them.
        outputs: The analysis's output names.
        inputs: The run's input paths.

    Raises:
        InvalidInvocation: For a missing or unknown output, a destination
            that is one of the inputs, an existing directory, shared by two
            outputs, under a file, or too long a name to stage.
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
    seen: dict[Path, str] = {}
    for name, path in sorted(destinations.items()):
        resolved = path.resolve()
        if resolved in seen:
            raise InvalidInvocation(
                f"outputs {seen[resolved]!r} and {name!r} have the same path {path}"
            )
        seen[resolved] = name
        _check_destination(name, path)


def _check_destination(name: str, path: Path) -> None:
    """Refuse a destination the staging can't write next to or replace.

    Args:
        name: The output name, for the message.
        path: Its destination.

    Raises:
        InvalidInvocation: If the path is a directory, too long a name to
            stage, or has a file where a folder should be.
    """
    if path.is_dir():
        raise InvalidInvocation(
            f"output {name!r} path {path} is a directory; give a file path"
        )
    if len(os.fsencode(_staging_name(path))) > _NAME_MAX:
        raise InvalidInvocation(f"output {name!r} file name is too long: {path.name}")
    ancestor = path.parent
    while not ancestor.exists() and ancestor != ancestor.parent:
        ancestor = ancestor.parent
    if not ancestor.is_dir():
        raise InvalidInvocation(
            f"output {name!r} path {path}: {ancestor} is not a directory"
        )
