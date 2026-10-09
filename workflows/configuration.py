"""Require complete resolved settings while keeping schemas local to each workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from workflow_execution.contracts import InvalidWorkflowInvocation


if TYPE_CHECKING:
    from collections.abc import Mapping


def validate_resolved_configuration[Configuration: BaseModel](
    schema: type[Configuration], value: Mapping[str, object], *, workflow_name: str
) -> Configuration:
    """Validate a fully resolved configuration rather than applying missing defaults.

    Args:
        schema: Workflow-owned immutable settings schema.
        value: Complete settings after catalog resolution.
        workflow_name: Safe descriptive name used in invocation errors.

    Returns:
        Configuration: Validated settings with every schema field explicitly supplied.

    Raises:
        InvalidWorkflowInvocation: Settings fail validation or omit declared fields.
    """
    try:
        config = schema.model_validate(value)
        if config.model_fields_set != schema.model_fields.keys():
            raise ValueError("incomplete configuration")
        return config
    except ValueError as error:
        raise InvalidWorkflowInvocation(
            f"resolved configuration does not match {workflow_name} workflow"
        ) from error
