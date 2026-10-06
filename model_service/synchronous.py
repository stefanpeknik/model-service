from __future__ import annotations

from typing import TYPE_CHECKING

from execution_scheduler.contracts import UnknownWorkflowError
from workflow_execution.application import validate_workflow_invocation
from workflow_execution.catalog import UnknownWorkflow
from workflow_execution.contracts import (
    InvalidWorkflowInvocation,
    WorkflowExecutionContext,
)


if TYPE_CHECKING:
    from workflow_execution.catalog import WorkflowCatalog
    from workflow_execution.contracts import WorkflowInvocation, WorkflowResult
    from workflow_execution.dispatch import WorkflowDispatch


class SynchronousWorkflowService:
    def __init__(
        self,
        catalog: WorkflowCatalog,
        dispatch: WorkflowDispatch,
    ) -> None:
        self._catalog = catalog
        self._dispatch = dispatch

    async def run(
        self,
        workflow_id: str,
        invocation: WorkflowInvocation,
    ) -> WorkflowResult:
        try:
            definition = self._catalog.get(workflow_id)
        except UnknownWorkflow as error:
            raise UnknownWorkflowError(workflow_id) from error
        validate_workflow_invocation(definition, invocation)
        try:
            self._catalog.validate_configuration(
                workflow_id,
                invocation.resolved_configuration,
            )
        except InvalidWorkflowInvocation:
            raise
        except (TypeError, ValueError) as error:
            raise InvalidWorkflowInvocation(str(error)) from error
        return await self._dispatch.run(
            workflow_id,
            WorkflowExecutionContext(),
            invocation,
        )
