"""What every pipeline run is resolved and checked against before it is dispatched or admitted.

Synchronous runs and scheduled admissions both go through `prepared_run`, so
the two accept and refuse the same requests.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from model_service.contracts.failures import InvalidInvocation


if TYPE_CHECKING:
    from model_service.catalog.pipelines import Pipeline, ResolvedPipeline
    from model_service.contracts.invocation import Invocation
    from model_service.control_plane.scheduler.public import PipelineResolver


def prepared_run(
    resolve: PipelineResolver, pipeline_id: str, invocation: Invocation
) -> tuple[ResolvedPipeline, Invocation]:
    """Resolve a pipeline with the invocation's settings and check the invocation against it.

    Args:
        resolve: Gives a deployed pipeline resolved with a run's settings.
        pipeline_id: The pipeline to run.
        invocation: The run; its configuration holds the caller's settings.

    Returns:
        The resolved pipeline, and the invocation recording every setting the
        run uses.

    Raises:
        UnknownPipelineError: If no deployed pipeline has this ID.
        InvalidInvocation: For an invalid setting, unknown or missing inputs, or
            `output_path` for a pipeline with several outputs.
    """
    resolved = resolve(pipeline_id, invocation.configuration)
    invocation = replace(invocation, configuration=resolved.pipeline.settings())
    _check_invocation(resolved.pipeline, invocation)
    return resolved, invocation


def _check_invocation(pipeline: Pipeline, invocation: Invocation) -> None:
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
