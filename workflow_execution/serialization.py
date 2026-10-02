from __future__ import annotations

from pydantic import TypeAdapter

from workflow_execution.contracts import WorkflowInvocation, WorkflowResult


_INVOCATION = TypeAdapter(WorkflowInvocation)
_RESULT = TypeAdapter(WorkflowResult)


def invocation_from_dict(value: object) -> WorkflowInvocation:
    return _INVOCATION.validate_python(value)


def invocation_to_dict(invocation: WorkflowInvocation) -> dict[str, object]:
    return _INVOCATION.dump_python(invocation, mode="json")


def result_to_dict(result: WorkflowResult) -> dict[str, object]:
    return _RESULT.dump_python(result, mode="json")


def result_from_dict(value: object) -> WorkflowResult:
    return _RESULT.validate_python(value)
