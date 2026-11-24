"""What every analysis run is checked against before it is dispatched or admitted."""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_service.contracts.failures import InvalidInvocation


if TYPE_CHECKING:
    from model_service.catalog.analyses import Analysis
    from model_service.contracts.invocation import Invocation


def check_invocation(analysis: Analysis, invocation: Invocation) -> None:
    """Refuse inputs or an output path the analysis cannot take.

    Args:
        analysis: The resolved analysis.
        invocation: The run's inputs and output path.

    Raises:
        InvalidInvocation: For unknown or missing inputs, or `output_path`
            for an analysis with several outputs.
    """
    inputs = ", ".join(sorted(analysis.inputs))
    if unknown := sorted(invocation.inputs.keys() - analysis.inputs.keys()):
        raise InvalidInvocation(
            f"unknown inputs: {', '.join(unknown)}; {analysis.id} reads {inputs}"
        )
    required = {name for name, needed in analysis.inputs.items() if needed}
    if missing := sorted(required - invocation.inputs.keys()):
        raise InvalidInvocation(f"missing inputs: {', '.join(missing)}")
    if invocation.output_path is not None and len(analysis.sinks) != 1:
        raise InvalidInvocation(
            f"output_path needs an analysis with one output; {analysis.id} has "
            f"{len(analysis.sinks)}"
        )
