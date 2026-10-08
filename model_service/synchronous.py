"""Validate and dispatch immediate workflows without creating durable execution records."""

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
    """Provide immediate workflow execution over a catalog and transport-independent dispatch adapter."""

    def __init__(
        self,
        catalog: WorkflowCatalog,
        dispatch: WorkflowDispatch,
    ) -> None:
        """Retain caller-owned catalog and dispatch dependencies without starting work.

        Args:
            catalog: Workflow definitions and their configuration validators/defaults.
            dispatch: Workflow transport used to invoke the selected Ray Serve
                application.
        """
        self._catalog = catalog
        self._dispatch = dispatch

    async def run(
        self,
        workflow_id: str,
        invocation: WorkflowInvocation,
    ) -> WorkflowResult:
        """Require a known workflow, valid inputs/deadline, and fully resolved configuration before dispatching with empty private context.

        Args:
            workflow_id: Declared workflow identifier used for catalog lookup and
                dispatch routing.
            invocation: Named inputs, resolved configuration, public access state, and
                absolute workflow deadline.

        Returns:
            WorkflowResult: Named inline outputs and/or references to completed artifact
                files.

        Raises:
            ConfirmedDispatchRejectionError: The remote deployment confirms rejection
                before accepting work.
            InvalidWorkflowInvocation: Inputs, resolved configuration, access context,
                or deadline violate the workflow contract.
            RemoteWorkflowFailureError: The remote workflow returns a safe structured
                failure.
            UnknownDispatchOutcomeError: Transport or response decoding cannot establish
                the remote outcome.
            UnknownWorkflowError: The workflow is not declared in the configured
                catalog.
        """
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
