"""Directories that the control plane and Ray replicas share, whatever users they run as.

A replica creates an attempt's workspace and outputs; the control plane later moves the
outputs or removes the workspace. The two may run as different users without a
shared group, so every directory either creates is open to all users (0777),
whatever the umask. Files keep their usual mode: removing or moving one needs
write access to its directory only.
"""

from __future__ import annotations

import logging
import shutil
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pathlib import Path


logger = logging.getLogger(__name__)


def make_shared_directory(path: Path) -> None:
    """Create a directory and its missing parents, each open to every user.

    The mode is set after creation, so the umask cannot narrow it. Existing
    directories keep their mode.

    Args:
        path: The directory to create.

    Raises:
        OSError: A directory could not be created or its mode set, or `path`
            exists and is not a directory.
    """
    missing = []
    current = path
    while not current.is_dir():
        missing.append(current)
        current = current.parent
    for directory in reversed(missing):
        try:
            directory.mkdir()
        except FileExistsError:
            # Created meanwhile by another run, which sets its mode itself; a
            # file in the way fails at the next level or below.
            if not directory.is_dir():
                raise
            continue
        directory.chmod(0o777)


def remove_directory(path: Path) -> None:
    """Remove a directory tree, logging a warning for what could not be removed.

    A missing directory is already removed. Removal never raises: what stays
    behind fills the shared volume, which the warning makes visible.

    Args:
        path: The directory to remove.
    """
    failures: list[BaseException] = []

    def record(_function: object, _path: str, error: BaseException) -> None:
        """Keep a failure, except for an entry already gone.

        Args:
            _function: The function that failed.
            _path: The entry it failed on.
            error: Why it failed.
        """
        if not isinstance(error, FileNotFoundError):
            failures.append(error)

    shutil.rmtree(path, onexc=record)
    if failures:
        logger.warning(
            "could not remove %d entries of %s, the first: %s",
            len(failures),
            path,
            failures[0],
        )
