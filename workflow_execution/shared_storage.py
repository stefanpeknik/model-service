from __future__ import annotations

from pathlib import Path, PurePosixPath

from workflow_execution.contracts import (
    InvalidWorkflowInvocation,
    SharedPathInput,
    WorkflowInput,
    WorkflowInvocation,
)


class SharedStorageAccess:
    def __init__(self, root: Path, inputs: dict[str, WorkflowInput]) -> None:
        try:
            self._root = root.resolve(strict=True)
        except OSError as error:
            raise InvalidWorkflowInvocation(
                "shared storage root is unavailable"
            ) from error
        if not self._root.is_dir():
            raise InvalidWorkflowInvocation("shared storage root must be a directory")
        self._inputs = inputs

    def input_path(self, name: str) -> Path:
        value = self._inputs.get(name)
        if not isinstance(value, SharedPathInput):
            raise InvalidWorkflowInvocation(f"input {name!r} is not a shared path")
        path = self._resolve(value.path, must_exist=True)
        if not path.is_file():
            raise InvalidWorkflowInvocation(
                f"input {name!r} must identify a regular file"
            )
        return path

    def output_path(self, relative_path: str) -> Path:
        return self._resolve(relative_path, must_exist=False)

    def is_visible_artifact(self, path: Path) -> bool:
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(self._root)
        except (OSError, ValueError):
            return False
        return resolved.is_file()

    def _resolve(self, relative_path: str, *, must_exist: bool) -> Path:
        path = PurePosixPath(relative_path)
        if (
            not relative_path
            or "\\" in relative_path
            or "\x00" in relative_path
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in relative_path.split("/"))
        ):
            raise InvalidWorkflowInvocation(
                "shared storage paths must be traversal-free relative paths"
            )
        candidate = self._root.joinpath(*path.parts)
        try:
            resolved = candidate.resolve(strict=must_exist)
            resolved.relative_to(self._root)
        except (OSError, ValueError) as error:
            raise InvalidWorkflowInvocation(
                "shared storage path is unavailable or escapes its root"
            ) from error
        return resolved


class SharedStorageAccessAdapter:
    def create(self, invocation: WorkflowInvocation) -> SharedStorageAccess:
        root = invocation.input_access.state.get("root")
        if not isinstance(root, str) or not root:
            raise InvalidWorkflowInvocation(
                "shared storage input access requires a root"
            )
        return SharedStorageAccess(Path(root), invocation.inputs)
