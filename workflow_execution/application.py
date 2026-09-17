from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Protocol

from workflow_execution.contracts import (
    InlineOutput,
    InvalidWorkflowInvocation,
    JsonValue,
    SharedArtifact,
    SharedPathInput,
    WorkflowContractViolation,
    WorkflowFailure,
    WorkflowInput,
    WorkflowInvocation,
    WorkflowResult,
    is_json_value,
)


if TYPE_CHECKING:
    from collections.abc import Awaitable, Mapping

    from workflow_execution.catalog import WorkflowDefinition


class Workflow(Protocol):
    async def run(self, context: WorkflowContext) -> WorkflowResult: ...


class WorkflowInputAccess(Protocol):
    def input_path(self, name: str) -> Path: ...

    def output_path(self, relative_path: str) -> Path: ...

    def is_visible_artifact(self, path: Path) -> bool: ...


class WorkflowInputAccessAdapter(Protocol):
    def create(
        self,
        invocation: WorkflowInvocation,
    ) -> WorkflowInputAccess | Awaitable[WorkflowInputAccess]: ...


@dataclass(frozen=True, slots=True)
class WorkflowContext:
    inputs: WorkflowInputAccess
    configuration: dict[str, JsonValue]
    deadline_at: datetime

    def input_path(self, name: str) -> Path:
        return self.inputs.input_path(name)

    def output_path(self, relative_path: str) -> Path:
        return self.inputs.output_path(relative_path)


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


class WorkflowApplication:
    def __init__(
        self,
        definition: WorkflowDefinition,
        workflow: Workflow,
        *,
        access_adapters: Mapping[str, WorkflowInputAccessAdapter] | None = None,
    ) -> None:
        self._definition = definition
        self._workflow = workflow
        self._access_adapters: dict[str, WorkflowInputAccessAdapter] = {
            "shared_storage": SharedStorageAccessAdapter(),
        }
        if access_adapters is not None:
            self._access_adapters.update(access_adapters)

    async def run(self, invocation: WorkflowInvocation) -> WorkflowResult:
        validate_workflow_invocation(self._definition, invocation)
        access = await self._create_access(invocation)
        context = WorkflowContext(
            inputs=access,
            configuration=invocation.resolved_configuration,
            deadline_at=invocation.deadline_at,
        )
        try:
            timeout_seconds = (
                invocation.deadline_at - datetime.now(UTC)
            ).total_seconds()
            async with asyncio.timeout(max(0, timeout_seconds)):
                result = await self._workflow.run(context)
            self._validate_result(result, access)
        except TimeoutError as error:
            raise WorkflowFailure(
                code="workflow_deadline_exceeded",
                safe_message="workflow deadline was exceeded",
                retryable=False,
            ) from error
        except (
            InvalidWorkflowInvocation,
            WorkflowContractViolation,
            WorkflowFailure,
        ):
            raise
        except Exception as error:
            raise WorkflowFailure(
                code="workflow_internal_error",
                safe_message="workflow failed unexpectedly",
                retryable=True,
            ) from error
        finally:
            close = getattr(access, "aclose", None)
            if close is not None:
                await close()
        return result

    async def _create_access(
        self,
        invocation: WorkflowInvocation,
    ) -> WorkflowInputAccess:
        adapter = self._access_adapters.get(invocation.input_access.kind)
        if adapter is None:
            raise InvalidWorkflowInvocation("unsupported input access kind")
        access = adapter.create(invocation)
        if inspect.isawaitable(access):
            return await access
        return access

    def _validate_result(
        self,
        result: WorkflowResult,
        access: WorkflowInputAccess,
    ) -> None:
        if not isinstance(result, WorkflowResult):
            raise WorkflowContractViolation("workflow returned an invalid result")
        if set(result.outputs) != set(self._definition.outputs):
            raise WorkflowContractViolation("outputs do not match workflow definition")
        for name, definition in self._definition.outputs.items():
            output = result.outputs[name]
            if not isinstance(output, InlineOutput | SharedArtifact):
                raise WorkflowContractViolation(
                    f"output {name!r} is not a workflow output"
                )
            if output.kind != definition.kind:
                raise WorkflowContractViolation(
                    f"output {name!r} does not match workflow definition"
                )
            if isinstance(output, SharedArtifact):
                path = Path(output.path)
                if not access.is_visible_artifact(path):
                    raise WorkflowContractViolation(
                        f"shared artifact {name!r} is not a visible completed file"
                    )
                if (
                    definition.media_type is not None
                    and output.media_type != definition.media_type
                ):
                    raise WorkflowContractViolation(
                        f"shared artifact {name!r} has the wrong media type"
                    )
            elif not is_json_value(output.value):
                raise WorkflowContractViolation(
                    f"inline output {name!r} is not JSON-compatible"
                )


def validate_workflow_invocation(
    definition: WorkflowDefinition,
    invocation: WorkflowInvocation,
    *,
    now: datetime | None = None,
) -> None:
    deadline = invocation.deadline_at
    if deadline.tzinfo is None or deadline.utcoffset() is None:
        raise InvalidWorkflowInvocation("deadline_at must include a timezone")
    if deadline <= (now or datetime.now(UTC)):
        raise InvalidWorkflowInvocation("deadline_at must be in the future")
    if set(invocation.inputs) != set(definition.inputs):
        raise InvalidWorkflowInvocation("inputs do not match workflow definition")
    for name, input_definition in definition.inputs.items():
        if invocation.inputs[name].kind not in input_definition.kinds:
            raise InvalidWorkflowInvocation(
                f"input {name!r} does not match workflow definition"
            )
