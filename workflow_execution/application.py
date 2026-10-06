from __future__ import annotations

import asyncio
import inspect
import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from integrations.shared_storage import SharedStorageAccessAdapter
from workflow_execution.cleanup import close_resource
from workflow_execution.contracts import (
    InlineOutput,
    InvalidWorkflowInvocation,
    JsonValue,
    SharedArtifact,
    WorkflowContractViolation,
    WorkflowExecutionContext,
    WorkflowFailure,
    WorkflowFailureCode,
    WorkflowInvocation,
    WorkflowResult,
    is_json_value,
)


logger = logging.getLogger(__name__)


if TYPE_CHECKING:
    from collections.abc import Awaitable, Mapping

    from workflow_execution.catalog import WorkflowDefinition


class Workflow(Protocol):
    async def run(self, context: WorkflowContext) -> WorkflowResult: ...


class WorkflowInputAccess(Protocol):
    def input_path(self, name: str) -> Path: ...

    def output_path(self, relative_path: str) -> Path: ...

    def is_visible_artifact(self, path: Path) -> bool: ...

    async def aclose(self) -> None: ...


class WorkflowInputAccessAdapter(Protocol):
    def create(
        self,
        invocation: WorkflowInvocation,
        execution_context: WorkflowExecutionContext,
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


class WorkflowApplication:
    def __init__(
        self,
        definition: WorkflowDefinition,
        workflow: Workflow,
        *,
        access_adapters: Mapping[str, WorkflowInputAccessAdapter] | None = None,
        cleanup_timeout_seconds: float = 10,
    ) -> None:
        if not math.isfinite(cleanup_timeout_seconds) or cleanup_timeout_seconds <= 0:
            raise ValueError("cleanup timeout must be finite and positive")
        self._definition = definition
        self._cleanup_timeout_seconds = cleanup_timeout_seconds
        self._workflow = workflow
        self._access_adapters: dict[str, WorkflowInputAccessAdapter] = {
            "shared_storage": SharedStorageAccessAdapter(),
        }
        if access_adapters is not None:
            self._access_adapters.update(access_adapters)

    async def run(
        self,
        invocation: WorkflowInvocation,
        execution_context: WorkflowExecutionContext | None = None,
    ) -> WorkflowResult:
        validate_workflow_invocation(self._definition, invocation)
        access: WorkflowInputAccess | None = None
        primary_failure: BaseException | None = None
        try:
            timeout_seconds = (
                invocation.deadline_at - datetime.now(UTC)
            ).total_seconds()
            timeout = asyncio.timeout(max(0, timeout_seconds))
            try:
                async with timeout:
                    access = await self._create_access(
                        invocation, execution_context or WorkflowExecutionContext()
                    )
                    context = WorkflowContext(
                        inputs=access,
                        configuration=invocation.resolved_configuration,
                        deadline_at=invocation.deadline_at,
                    )
                    result = await self._workflow.run(context)
                    self._validate_result(result, access)
            except TimeoutError as error:
                if not timeout.expired():
                    raise
                raise WorkflowFailure(
                    code=WorkflowFailureCode.DEADLINE_EXCEEDED,
                    safe_message="workflow deadline was exceeded",
                    retryable=False,
                ) from error
        except (
            InvalidWorkflowInvocation,
            WorkflowContractViolation,
            WorkflowFailure,
        ) as error:
            primary_failure = error
            raise
        except Exception as error:
            primary_failure = WorkflowFailure(
                code=WorkflowFailureCode.INTERNAL_ERROR,
                safe_message="workflow failed unexpectedly",
                retryable=True,
            )
            raise primary_failure from error
        except BaseException as error:
            primary_failure = error
            raise
        finally:
            if access is not None:
                try:
                    await close_resource(
                        access.aclose,
                        timeout_seconds=self._cleanup_timeout_seconds,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    if primary_failure is None:
                        raise WorkflowFailure(
                            code=WorkflowFailureCode.INTERNAL_ERROR,
                            safe_message="workflow resource cleanup failed",
                            retryable=True,
                        ) from error
                    logger.warning(
                        "workflow resource cleanup failed: %s", type(error).__name__
                    )
        return result

    async def _create_access(
        self,
        invocation: WorkflowInvocation,
        execution_context: WorkflowExecutionContext,
    ) -> WorkflowInputAccess:
        adapter = self._access_adapters.get(invocation.input_access.kind)
        if adapter is None:
            raise InvalidWorkflowInvocation("unsupported input access kind")
        access = adapter.create(invocation, execution_context)
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
