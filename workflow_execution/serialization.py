"""Validate and serialize workflow invocations and results for HTTP and JSONB storage."""

from __future__ import annotations

from pydantic import TypeAdapter

from workflow_execution.contracts import WorkflowInvocation, WorkflowResult


_INVOCATION = TypeAdapter(WorkflowInvocation)
_RESULT = TypeAdapter(WorkflowResult)


def invocation_from_dict(value: object) -> WorkflowInvocation:
    """Validate untrusted decoded data as a typed workflow invocation.

    Args:
        value: Untrusted decoded invocation object from HTTP or stored JSONB.

    Returns:
        WorkflowInvocation: Validated typed invocation with explicitly tagged inputs and
            parsed deadline.

    Raises:
        ValidationError: Decoded fields or model configuration do not satisfy the typed
            schema.
    """
    return _INVOCATION.validate_python(value)


def invocation_to_dict(invocation: WorkflowInvocation) -> dict[str, object]:
    """Serialize an invocation to JSON-compatible values, including its ISO datetime and tagged inputs.

    Args:
        invocation: Named inputs, resolved configuration, public access state, and
            absolute workflow deadline.

    Returns:
        dict[str, object]: JSON-compatible invocation fields with tagged inputs and an
            ISO deadline, excluding private execution context.
    """
    return _INVOCATION.dump_python(invocation, mode="json")


def result_to_dict(result: WorkflowResult) -> dict[str, object]:
    """Serialize typed output values and artifact references to JSON-compatible data.

    Args:
        result: Typed workflow outputs to validate or serialize.

    Returns:
        dict[str, object]: JSON-compatible named inline values and artifact references.
    """
    return _RESULT.dump_python(result, mode="json")


def result_from_dict(value: object) -> WorkflowResult:
    """Validate decoded output data as a typed workflow result.

    Args:
        value: Untrusted decoded result object from HTTP or stored JSONB.

    Returns:
        WorkflowResult: Validated named inline outputs and artifact references.

    Raises:
        ValidationError: Decoded fields or model configuration do not satisfy the typed
            schema.
    """
    return _RESULT.validate_python(value)
