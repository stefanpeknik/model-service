"""What every pipeline run is checked against before it is dispatched or admitted."""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_service.contracts.failures import InvalidInvocation


if TYPE_CHECKING:
    from model_service.catalog.pipelines import Pipeline
    from model_service.contracts.invocation import Invocation


def check_invocation(pipeline: Pipeline, invocation: Invocation) -> None:
    """Refuse inputs or an output path the pipeline cannot take.

    Args:
        pipeline: The resolved pipeline.
        invocation: The run's inputs and output path.

    Raises:
        InvalidInvocation: For unknown or missing inputs, or `output_path`
            for a pipeline with several outputs.
    """
    inputs = ", ".join(sorted(pipeline.inputs))
    if unknown := sorted(invocation.inputs.keys() - pipeline.inputs.keys()):
        raise InvalidInvocation(
            f"unknown inputs: {', '.join(unknown)}; {pipeline.id} reads {inputs}"
        )
    required = {name for name, needed in pipeline.inputs.items() if needed}
    if missing := sorted(required - invocation.inputs.keys()):
        raise InvalidInvocation(f"missing inputs: {', '.join(missing)}")
    if invocation.output_path is not None and len(pipeline.sinks) != 1:
        raise InvalidInvocation(
            f"output_path needs a pipeline with one output; {pipeline.id} has "
            f"{len(pipeline.sinks)}"
        )
