"""Expose provider-neutral workflow invocation and output values.

application.py owns computation lifetime, catalog.py declares contracts/configuration,
dispatch.py crosses the internal HTTP boundary, and cleanup.py closes acquired
resources.
"""

from workflow_execution.application import WorkflowApplication
from workflow_execution.contracts import WorkflowInvocation, WorkflowResult


__all__ = ["WorkflowApplication", "WorkflowInvocation", "WorkflowResult"]
