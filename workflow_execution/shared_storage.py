from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from workflow_execution.contracts import (
    InvalidWorkflowInvocation,
    SharedPathInput,
    WorkflowExecutionContext,
    WorkflowInput,
    WorkflowInvocation,
)


class SharedStorageState(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    root: Annotated[str, Field(min_length=1)]


class SharedStorageAccess:
    def __init__(
        self,
        root: Path,
        inputs: dict[str, WorkflowInput],
        *,
        processing_attempt_id: str | None = None,
    ) -> None:
        try:
            self._root = root.resolve(strict=True)
        except OSError as error:
            raise InvalidWorkflowInvocation(
                "shared storage root is unavailable"
            ) from error
        if not self._root.is_dir():
            raise InvalidWorkflowInvocation("shared storage root must be a directory")
        self._inputs = inputs
        self._output_prefix = (
            ".workflow-attempts/"
            + hashlib.sha256(processing_attempt_id.encode()).hexdigest()
            + "/"
            if processing_attempt_id is not None
            else ""
        )

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
        self,
        relative_path: str,
        *,
        must_exist: bool,
        prefix: str = "",
    ) -> Path:
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
        candidate = self._root.joinpath(*PurePosixPath(prefix).parts, *path.parts)
        try:
            resolved = candidate.resolve(strict=must_exist)
            resolved.relative_to(self._root)
        except (OSError, ValueError) as error:
            raise InvalidWorkflowInvocation(
                "shared storage path is unavailable or escapes its root"
            ) from error
        return resolved


class SharedStorageAccessAdapter:
    def create(
        self,
        invocation: WorkflowInvocation,
        execution_context: WorkflowExecutionContext,
    ) -> SharedStorageAccess:
        try:
            state = SharedStorageState.model_validate(invocation.input_access.state)
        except ValidationError as error:
            raise InvalidWorkflowInvocation(
                "shared storage input access requires only a non-empty root"
            ) from error
        return SharedStorageAccess(
            Path(state.root),
            invocation.inputs,
            processing_attempt_id=execution_context.processing_attempt_id,
        )
