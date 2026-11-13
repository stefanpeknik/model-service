"""Resolve builder files below a shared-storage root with traversal and symlink checks.

The caller names the root, but it must lie within a root the server allows.
Scheduled attempts write outputs below `.attempts/<execution>/<hash of attempt
token>/`, and the provider's finish moves the accepted outputs to their
requested paths; synchronous runs write to the requested path directly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from model_service.builders.input_access import InputAccess, InputAccessFactory
from model_service.contracts.failures import InvalidInvocation
from model_service.contracts.invocation import SharedPathInput


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from model_service.contracts.invocation import (
        BuilderInput,
        ExecutionContext,
        Invocation,
    )
    from model_service.contracts.model_declarations import ModelBinding


ATTEMPTS_DIR = ".attempts"
# Protected-context key naming the execution's directory below ATTEMPTS_DIR.
ATTEMPTS_KEY = "attempts"


class SharedStorageState(BaseModel):
    """Input access state of a shared-storage invocation.

    Attributes:
        root: The directory inputs and outputs are relative to.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    root: Annotated[str, Field(min_length=1)]


class SharedStorageAccess(InputAccess):
    """Resolves inputs and outputs below one checked shared-storage root."""

    def __init__(
        self,
        root: Path,
        inputs: dict[str, BuilderInput],
        *,
        output_prefix: str = "",
    ) -> None:
        """Hold the resolved root, the inputs and the attempt's output prefix.

        Args:
            root: The resolved shared-storage root, already checked against the allowed roots.
            inputs: The invocation's builder inputs by name.
            output_prefix: Relative directory a scheduled attempt writes below; empty
                for synchronous runs, which write to the requested path directly.
        """
        self._root = root
        self._inputs = inputs
        self._output_prefix = output_prefix

    def input_path(self, name: str) -> Path:
        """Resolve a shared-path input to an existing regular file below the root.

        Args:
            name: The builder input name.

        Returns:
            The resolved file path.

        Raises:
            InvalidInvocation: If the input is not a shared path, its path is not
                relative and traversal-free, it is missing or escapes the root, or it
                is not a regular file.
        """
        value = self._inputs.get(name)
        if not isinstance(value, SharedPathInput):
            raise InvalidInvocation(f"input {name!r} is not a shared path")
        path = self._resolve(value.path, must_exist=True)
        if not path.is_file():
            raise InvalidInvocation(f"input {name!r} must be a regular file")
        return path

    def output_path(self, relative_path: str) -> Path:
        """Resolve an output path below the root and the attempt's prefix.

        The file need not exist yet.

        Args:
            relative_path: The relative POSIX path the builder writes to.

        Returns:
            The resolved output path.

        Raises:
            InvalidInvocation: If the path is not relative and traversal-free, or
                resolves outside the root.
        """
        return self._resolve(
            relative_path, must_exist=False, prefix=self._output_prefix
        )

    async def aclose(self) -> None:
        """Release nothing; shared-storage access holds no resources."""
        return None

    def is_visible_artifact(self, path: Path) -> bool:
        """Check that a path is an existing file below the root once symlinks are resolved.

        Args:
            path: The artifact path a builder reported.

        Returns:
            True if the path resolves to a file inside the root.
        """
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(self._root)
        except (OSError, ValueError):
            return False
        return resolved.is_file()

    def _resolve(
        self, relative_path: str, *, must_exist: bool, prefix: str = ""
    ) -> Path:
        """Resolve a relative POSIX path below the root, rejecting traversal and symlink escapes.

        Args:
            relative_path: The caller's relative path.
            must_exist: Whether the path must already exist.
            prefix: A relative directory placed between the root and the path.

        Returns:
            The resolved path.

        Raises:
            InvalidInvocation: If the path is absolute or contains a backslash, NUL,
                or an empty, `.` or `..` part; or if it is missing when it must exist,
                or resolves outside the root.
        """
        path = PurePosixPath(relative_path)
        if (
            "\\" in relative_path
            or "\x00" in relative_path
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in relative_path.split("/"))
        ):
            raise InvalidInvocation(
                "shared-storage paths must be relative and traversal-free"
            )
        candidate = self._root.joinpath(*PurePosixPath(prefix).parts, *path.parts)
        try:
            resolved = candidate.resolve(strict=must_exist)
            resolved.relative_to(self._root)
        except (OSError, ValueError) as error:
            raise InvalidInvocation(
                "shared-storage path is unavailable or escapes its root"
            ) from error
        return resolved


class SharedStorageAccessFactory(InputAccessFactory):
    """Creates shared-storage access for roots within the allowed ones."""

    def __init__(self, allowed_roots: Iterable[Path]) -> None:
        """Keep the roots callers may name.

        Args:
            allowed_roots: Directories a requested root must lie within.
        """
        self._allowed_roots = tuple(allowed_roots)

    def root(self, state: Mapping[str, object]) -> Path:
        """Validate and resolve the root an invocation's input access state names.

        Args:
            state: The input access state; it must hold only `root`.

        Returns:
            The resolved root directory.

        Raises:
            InvalidInvocation: If the state has fields other than `root`, or the root
                does not exist, is not a directory, or lies outside every allowed root.
        """
        try:
            requested = SharedStorageState.model_validate(state).root
        except ValidationError as error:
            raise InvalidInvocation(
                "shared-storage access requires only a root"
            ) from error
        try:
            root = Path(requested).resolve(strict=True)
        except OSError as error:
            raise InvalidInvocation("shared-storage root is unavailable") from error
        if not root.is_dir():
            raise InvalidInvocation("shared-storage root must be a directory")
        for allowed in self._allowed_roots:
            try:
                root.relative_to(allowed.resolve())
            except (OSError, ValueError):
                continue
            return root
        raise InvalidInvocation("shared-storage root is not allowed on this server")

    def create(
        self,
        invocation: Invocation,
        execution_context: ExecutionContext,
        model: ModelBinding | None = None,
    ) -> SharedStorageAccess:
        """Create shared-storage access for one invocation.

        Args:
            invocation: The invocation; its input access state names the root.
            execution_context: Supplies the processing attempt ID and the
                execution's attempts directory that scope outputs.
            model: Unused.

        Returns:
            Access to the invocation's inputs and outputs below the root.

        Raises:
            InvalidInvocation: If the root is invalid or not allowed on this server.
        """
        del model
        attempt = execution_context.processing_attempt_id
        group = execution_context.protected_context.get(ATTEMPTS_KEY)
        prefix = ""
        if attempt is not None:
            prefix = ATTEMPTS_DIR + (f"/{group}" if isinstance(group, str) else "")
            prefix += "/" + hashlib.sha256(attempt.encode()).hexdigest()
        return SharedStorageAccess(
            self.root(invocation.input_access.state),
            invocation.inputs,
            output_prefix=prefix,
        )
