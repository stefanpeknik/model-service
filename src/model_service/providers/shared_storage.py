"""Resolve builder files below a shared-storage root with traversal and symlink checks.

The caller names the root, but it must lie within a root the server allows.
Scheduled attempts write outputs below `.attempts/<hash of attempt token>/`;
synchronous runs write to the requested relative path directly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from model_service.builders.contracts import InvalidInvocation, SharedPathInput


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from model_service.builders.contracts import (
        BuilderInput,
        ExecutionContext,
        Invocation,
    )


class SharedStorageState(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    root: Annotated[str, Field(min_length=1)]


class SharedStorageAccess:
    def __init__(
        self,
        root: Path,
        inputs: dict[str, BuilderInput],
        *,
        processing_attempt_id: str | None = None,
    ) -> None:
        self._root = root
        self._inputs = inputs
        self._output_prefix = (
            ".attempts/" + hashlib.sha256(processing_attempt_id.encode()).hexdigest()
            if processing_attempt_id is not None
            else ""
        )

    def input_path(self, name: str) -> Path:
        value = self._inputs.get(name)
        if not isinstance(value, SharedPathInput):
            raise InvalidInvocation(f"input {name!r} is not a shared path")
        path = self._resolve(value.path, must_exist=True)
        if not path.is_file():
            raise InvalidInvocation(f"input {name!r} must be a regular file")
        return path

    def output_path(self, relative_path: str) -> Path:
        return self._resolve(
            relative_path, must_exist=False, prefix=self._output_prefix
        )

    async def aclose(self) -> None:
        return None

    def is_visible_artifact(self, path: Path) -> bool:
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(self._root)
        except (OSError, ValueError):
            return False
        return resolved.is_file()

    def _resolve(
        self, relative_path: str, *, must_exist: bool, prefix: str = ""
    ) -> Path:
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


class SharedStorageAccessAdapter:
    def __init__(self, allowed_roots: Iterable[Path]) -> None:
        self._allowed_roots = tuple(allowed_roots)

    def root(self, state: Mapping[str, object]) -> Path:
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
        self, invocation: Invocation, execution_context: ExecutionContext
    ) -> SharedStorageAccess:
        return SharedStorageAccess(
            self.root(invocation.input_access.state),
            invocation.inputs,
            processing_attempt_id=execution_context.processing_attempt_id,
        )
