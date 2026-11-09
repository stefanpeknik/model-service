"""Resolve builder files below a shared-storage root with traversal and symlink checks.

The caller names the root, but it must lie within a root the server allows.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from model_service.builders.access import AccessAdapter, InputAccess
from model_service.builders.contracts import InvalidInvocation, SharedPathInput


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from model_service.builders.contracts import (
        BuilderInput,
        Invocation,
        ModelBinding,
    )


class SharedStorageState(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    root: Annotated[str, Field(min_length=1)]


class SharedStorageAccess(InputAccess):
    def __init__(self, root: Path, inputs: dict[str, BuilderInput]) -> None:
        """Hold the resolved root and the inputs.

        Args:
            root: The resolved shared-storage root, already checked against the allowed roots.
            inputs: The invocation's builder inputs by name.
        """
        self._root = root
        self._inputs = inputs

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
        """Resolve an output path below the root.

        The file need not exist yet.

        Args:
            relative_path: The relative POSIX path the builder writes to.

        Returns:
            The resolved output path.

        Raises:
            InvalidInvocation: If the path is not relative and traversal-free, or
                resolves outside the root.
        """
        return self._resolve(relative_path, must_exist=False)

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

    def _resolve(self, relative_path: str, *, must_exist: bool) -> Path:
        """Resolve a relative POSIX path below the root, rejecting traversal and symlink escapes.

        Args:
            relative_path: The caller's relative path.
            must_exist: Whether the path must already exist.

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
        candidate = self._root.joinpath(*path.parts)
        try:
            resolved = candidate.resolve(strict=must_exist)
            resolved.relative_to(self._root)
        except (OSError, ValueError) as error:
            raise InvalidInvocation(
                "shared-storage path is unavailable or escapes its root"
            ) from error
        return resolved


class SharedStorageAccessAdapter(AccessAdapter):
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
        model: ModelBinding | None = None,
    ) -> SharedStorageAccess:
        """Create shared-storage access for one invocation.

        Args:
            invocation: The invocation; its input access state names the root.
            model: Unused.

        Returns:
            Access to the invocation's inputs and outputs below the root.

        Raises:
            InvalidInvocation: If the root is invalid or not allowed on this server.
        """
        del model
        return SharedStorageAccess(
            self.root(invocation.input_access.state), invocation.inputs
        )
