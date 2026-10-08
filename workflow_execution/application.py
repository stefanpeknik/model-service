"""Execute a workflow behind input validation, deadline enforcement, and resource cleanup.

Access adapters translate integration-specific data into paths. Workflows receive
WorkflowContext and return typed results without owning HTTP or scheduler state.
"""

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
    """Define the computation hook implemented by each workflow."""

    async def run(self, context: WorkflowContext) -> WorkflowResult:
        """Compute declared outputs from prepared input access and configuration within the supplied deadline.

        Args:
            context: Invocation-scoped input/output access, resolved configuration, and
                absolute deadline.

        Returns:
            WorkflowResult: Named inline outputs and/or references to completed artifact
                files.
        """
        ...


class WorkflowInputAccess(Protocol):
    """Provide invocation-scoped input/output paths, artifact visibility checks, and cleanup."""

    def input_path(self, name: str) -> Path:
        """Resolve a named input to a local readable path or reject unsupported input access.

        Args:
            name: Declared workflow input name to resolve.

        Returns:
            Path: Existing local file for the named input; invalid or unavailable inputs
                raise an invocation error.

        Raises:
            InvalidWorkflowInvocation: The named input cannot be resolved through this
                access adapter.
        """
        ...

    def output_path(self, relative_path: str) -> Path:
        """Allocate an output path within this invocation's permitted storage root.

        Args:
            relative_path: Output path relative to the permitted root.

        Returns:
            Path: Permitted output destination; allocating the path does not itself
                create or complete the file.
        """
        ...

    def is_visible_artifact(self, path: Path) -> bool:
        """Check that an artifact is a completed file visible to the caller of this invocation.

        Args:
            path: Completed artifact path to check within this invocation's storage
                boundary.

        Returns:
            bool: True only for a completed regular file visible within this
                invocation's storage root.
        """
        ...

    async def aclose(self) -> None:
        """Release invocation-scoped sessions and temporary input resources.

        Returns:
            None: No return value.
        """
        ...


class WorkflowInputAccessAdapter(Protocol):
    """Create input access for one invocation using public and protected execution context."""

    def create(
        self,
        invocation: WorkflowInvocation,
        execution_context: WorkflowExecutionContext,
    ) -> WorkflowInputAccess | Awaitable[WorkflowInputAccess]:
        """Construct input access synchronously or asynchronously, materializing provider data when necessary.

        Args:
            invocation: Named inputs, resolved configuration, public access state, and
                absolute workflow deadline.
            execution_context: Private attempt identity and provider context kept
                separate from the invocation.

        Returns:
            WorkflowInputAccess | Awaitable[WorkflowInputAccess]: Created invocation
                access, or an awaitable that resolves to it.
        """
        ...


@dataclass(frozen=True, slots=True)
class WorkflowContext:
    """Expose input/output access, resolved configuration, and the deadline to provider-neutral workflow code.

    Attributes:
        inputs: Input names mapped to the invocation's typed input values.
        configuration: Fully resolved JSON workflow settings.
        deadline_at: Absolute aware deadline, including input acquisition and execution.
    """

    inputs: WorkflowInputAccess
    configuration: dict[str, JsonValue]
    deadline_at: datetime

    def input_path(self, name: str) -> Path:
        """Resolve a declared input name through the active access adapter.

        Args:
            name: Declared workflow input name to resolve.

        Returns:
            Path: Existing local file for the named input; invalid or unavailable inputs
                raise an invocation error.

        Raises:
            InvalidWorkflowInvocation: The selected access adapter cannot resolve the
                named input.
        """
        return self.inputs.input_path(name)

    def output_path(self, relative_path: str) -> Path:
        """Allocate a workflow output path through the active access adapter.

        Args:
            relative_path: Output path relative to the permitted root.

        Returns:
            Path: Permitted output destination; allocating the path does not itself
                create or complete the file.

        Raises:
            InvalidWorkflowInvocation: The selected access adapter rejects the output
                path.
        """
        return self.inputs.output_path(relative_path)


class WorkflowApplication:
    """Wrap computation with contract checks, absolute deadlines, safe failures, and bounded cleanup."""

    def __init__(
        self,
        definition: WorkflowDefinition,
        workflow: Workflow,
        *,
        access_adapters: Mapping[str, WorkflowInputAccessAdapter] | None = None,
        cleanup_timeout_seconds: float = 10,
    ) -> None:
        """Register access adapters, including shared storage, and require a finite positive cleanup timeout.

        Args:
            definition: Workflow input/output contract or loaded YAML mapping to
                translate.
            workflow: Provider-neutral computation implementation wrapped by
                WorkflowApplication.
            access_adapters: Additional access-kind adapters; shared storage is
                registered by default.
            cleanup_timeout_seconds: Maximum wait for closing invocation-owned
                resources.

        Raises:
            ValueError: cleanup_timeout_seconds is non-finite or nonpositive.
        """
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
        """Validate the invocation, create access, execute under its deadline, validate outputs, and close access.

        Unexpected errors become retryable safe failures. Cleanup runs even on
        cancellation; cleanup errors preserve an existing primary failure but fail an
        otherwise successful run.

        Args:
            invocation: Named inputs, resolved configuration, public access state, and
                absolute workflow deadline.
            execution_context: Private attempt identity and provider context kept
                separate from the invocation.

        Returns:
            WorkflowResult: Named inline outputs and/or references to completed artifact
                files.

        Raises:
            InvalidWorkflowInvocation: Inputs, resolved configuration, access context,
                or deadline violate the workflow contract.
            WorkflowContractViolation: The computation returned invalid or incomplete
                declared outputs.
            WorkflowFailure: Execution or resource cleanup fails with a safe classified
                workflow error.
            asyncio.CancelledError: The owning task, caller, deadline, or lease monitor
                cancels the operation.
        """
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
        """Select the invocation's access adapter, support sync or async creation, and reject an unknown kind.

        Args:
            invocation: Named inputs, resolved configuration, public access state, and
                absolute workflow deadline.
            execution_context: Private attempt identity and provider context kept
                separate from the invocation.

        Returns:
            WorkflowInputAccess: Invocation-scoped paths, visibility checks, and cleanup
                boundary.

        Raises:
            InvalidWorkflowInvocation: Inputs, resolved configuration, access context,
                or deadline violate the workflow contract.
        """
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
        """Require exactly the declared outputs, matching kinds/media types, finite inline JSON, and completed visible artifact files.

        Args:
            result: Typed workflow outputs to validate or serialize.
            access: Invocation-scoped file access used to verify artifact visibility.

        Returns:
            None: No return value.

        Raises:
            WorkflowContractViolation: The computation returned invalid or incomplete
                declared outputs.
        """
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
    """Require a future timezone-aware deadline and exactly the declared input names with accepted kinds.

    Args:
        definition: Workflow input/output contract or loaded YAML mapping to translate.
        invocation: Named inputs, resolved configuration, public access state, and
            absolute workflow deadline.
        now: Aware current time supplied by the repository or validation clock.

    Returns:
        None: No return value.

    Raises:
        InvalidWorkflowInvocation: Inputs, resolved configuration, access context, or
            deadline violate the workflow contract.
    """
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
