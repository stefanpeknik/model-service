"""Resolve workflow files under a shared storage root with traversal and symlink checks.

Scheduled attempts isolate outputs under a hash of their processing token. Immediate
executions use the requested relative output path directly.
"""

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
    """Require exactly one nonempty shared storage root in the public access context.

    Attributes:
        root: Shared storage directory visible to the workflow process.
    """

    model_config = ConfigDict(strict=True, frozen=True, extra="forbid")

    root: Annotated[str, Field(min_length=1)]


class SharedStorageAccess:
    """Expose readable inputs and permitted output paths beneath a resolved shared directory."""

    def __init__(
        self,
        root: Path,
        inputs: dict[str, WorkflowInput],
        *,
        processing_attempt_id: str | None = None,
    ) -> None:
        """Require an existing directory root and derive an optional isolated output prefix for the processing attempt.

        Args:
            root: Existing local directory containing permitted input/output storage.
            inputs: Declared input names and their inline/path/provider references.
            processing_attempt_id: Optional attempt token used to isolate scheduled
                output directories.

        Raises:
            InvalidWorkflowInvocation: Inputs, resolved configuration, access context,
                or deadline violate the workflow contract.
        """
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
        """Resolve a named shared-path input and require a regular file contained within the root.

        Args:
            name: Declared workflow input name to resolve.

        Returns:
            Path: Existing local file for the named input; invalid or unavailable inputs
                raise an invocation error.

        Raises:
            InvalidWorkflowInvocation: Inputs, resolved configuration, access context,
                or deadline violate the workflow contract.
        """
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
        """Resolve a traversal-free output path, adding the attempt prefix when configured; the file need not exist yet.

        Args:
            relative_path: Output path relative to the permitted root.

        Returns:
            Path: Permitted output destination; allocating the path does not itself
                create or complete the file.

        Raises:
            InvalidWorkflowInvocation: Inputs, resolved configuration, access context,
                or deadline violate the workflow contract.
        """
        return self._resolve(
            relative_path, must_exist=False, prefix=self._output_prefix
        )

    async def aclose(self) -> None:
        """Complete without cleanup because shared storage access owns no sessions or temporary resources.

        Returns:
            None: No return value.
        """
        return None

    def is_visible_artifact(self, path: Path) -> bool:
        """Accept only an existing regular file whose resolved path stays within the shared root.

        Args:
            path: Completed artifact path to check within this invocation's storage
                boundary.

        Returns:
            bool: True only for a completed regular file visible within this
                invocation's storage root.
        """
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
        """Validate a relative POSIX path and ensure its resolved target stays within the root, including symlink resolution.

        Args:
            relative_path: Output path relative to the permitted root.
            must_exist: Whether resolution requires the path to exist already.
            prefix: Internal relative output prefix added before the caller's path.

        Returns:
            Path: Resolved local file or output destination within the permitted storage
                root.

        Raises:
            InvalidWorkflowInvocation: Inputs, resolved configuration, access context,
                or deadline violate the workflow contract.
        """
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
    """Create shared storage access from validated invocation state and private attempt identity."""

    def create(
        self,
        invocation: WorkflowInvocation,
        execution_context: WorkflowExecutionContext,
    ) -> SharedStorageAccess:
        """Validate the root-only access context and construct access with attempt-isolated outputs when scheduled.

        Args:
            invocation: Named inputs, resolved configuration, public access state, and
                absolute workflow deadline.
            execution_context: Private attempt identity and provider context kept
                separate from the invocation.

        Returns:
            SharedStorageAccess: Validated shared root access with optional isolated
                attempt outputs.

        Raises:
            InvalidWorkflowInvocation: Inputs, resolved configuration, access context,
                or deadline violate the workflow contract.
        """
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
