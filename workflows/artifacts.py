"""Publish closed output files without exposing partial artifacts."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import TYPE_CHECKING
from uuid import uuid4

from workflow_execution.contracts import InvalidWorkflowInvocation


if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@contextmanager
def atomic_artifact(destination: Path, *, inputs: tuple[Path, ...]) -> Iterator[Path]:
    """Yield a sibling temporary path and replace the destination only on success.

    Args:
        destination: Final output path resolved by invocation-scoped storage access.
        inputs: Resolved input paths which publication must not replace.

    Yields:
        Path: Unique temporary path; the caller must close its writer before exit.

    Raises:
        InvalidWorkflowInvocation: The destination aliases an input path.
        RuntimeError: Successful computation did not produce a regular file.
        OSError: Creating directories, publishing, or removing the temporary file fails.
    """
    if any(destination.resolve() == source.resolve() for source in inputs):
        raise InvalidWorkflowInvocation("output must not replace a workflow input")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.{uuid4().hex}{destination.suffix}"
    )
    try:
        yield temporary
        if not temporary.is_file():
            raise RuntimeError("workflow did not produce a regular artifact file")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
