"""Pipeline files: resolving one with every load-time check, and their schema.

A pipeline file names a source, steps (which may call models) and
sinks:

```yaml
summary: Scores each tile of a prostate slide into a heatmap.
source: {use: slide_tiles, stride_fraction: 0.5}
steps:
  - {model: prostate-classifier-1}
sinks:
  - {use: score_grid, output: heatmap}
```
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ConfigDict, Field, create_model

from components import slots
from model_service.catalog.pipelines import checks
from model_service.catalog.pipelines.declaration import Pipeline
from model_service.catalog.pipelines.model_step import ModelStep
from model_service.catalog.pipelines.providers import (
    providers_schema,
    resolve_providers,
)
from model_service.catalog.schema import usable


if TYPE_CHECKING:
    from components import Sink, Source, Step
    from model_service.catalog.context import ResolveContext
    from model_service.catalog.errors import YamlPath
    from model_service.catalog.pipelines.declaration import PipelineStep
    from model_service.registry import Registry


FOLDER = "pipelines"
SUMMARY = "Pipeline files: a source, steps (which may call models) and sinks."


class Fields(BaseModel):
    """Every top-level key of a pipeline file.

    The component keys are resolved through the registry; here they are only
    listed so a misspelt key names every key that fits.

    Attributes:
        summary: One line saying what the pipeline does.
        source: The source.
        steps: Steps, in order.
        sinks: Sinks.
        providers: How the pipeline is offered beyond the public API, e.g. in EMPAIA.
    """

    model_config = ConfigDict(extra="forbid")

    summary: str = Field("", description="One line saying what the pipeline does.")
    source: Any = Field(None, description="A source: `{use: slide_tiles, …}`.")
    steps: Any = Field(
        None, description="Steps in order: `{model: <id>}` or `{use: <step>, …}`."
    )
    sinks: Any = Field(None, description="Sinks: `[{use: score_grid, output: …}, …]`.")
    providers: Any = Field(
        None, description="How the pipeline is offered: `{empaia: {…}}` for an app."
    )


class ModelStepFields(BaseModel):
    """A model step: calls a model on every tile."""

    model_config = ConfigDict(extra="forbid", title="model step")

    model: str = Field(description="A model ID, the name of a file in `models/`.")


def build_pipeline(document: dict[str, Any], ctx: ResolveContext) -> Pipeline | None:
    """Resolve the components of one pipeline file and run every load-time check.

    Args:
        document: The pipeline file after the environment's patch.
        ctx: Resolves components and models and collects errors.

    Returns:
        The `Pipeline`, or None if anything failed.
    """
    fields = ctx.fields(Fields, document)
    source = ctx.component(slots.SOURCE, document.get("source"), ("source",))
    steps = _steps(ctx, document.get("steps"))
    sinks = ctx.components(
        slots.SINKS, document.get("sinks"), ("sinks",), required=True
    )
    if fields is None or source is None or steps is None or ctx.errors:
        return None
    sink_list = cast("list[Sink]", sinks)
    filled = checks.fill_geometry(ctx, cast("Source", source), steps)
    columns = checks.fold_columns(ctx, filled, steps, sink_list)
    checks.check_settings(ctx, filled, steps, sink_list)
    checks.check_outputs(ctx, sink_list)
    if ctx.errors:
        return None
    pipeline = Pipeline(
        id=ctx.document_id,
        summary=fields.summary,
        source=filled,
        steps=tuple(steps),
        sinks=tuple(sink_list),
        inputs=filled.inputs(),
        columns=columns,
        declaration=document,
    )
    empaia = resolve_providers(ctx, document.get("providers"), pipeline)
    if ctx.errors:
        return None
    return replace(pipeline, empaia=empaia)


def pipeline_schema(registry: Registry) -> type[BaseModel]:
    """A pipeline file, with the sources, steps and sinks there are.

    Args:
        registry: The components files may name.

    Returns:
        The file's model, for its JSON Schema.
    """
    step = usable(registry, slots.STEPS) | ModelStepFields
    sink = usable(registry, slots.SINKS)
    info = Fields.model_fields
    return create_model(
        "pipeline",
        __base__=Fields,
        __doc__=SUMMARY,
        source=(
            usable(registry, slots.SOURCE),
            Field(description=info["source"].description),
        ),
        steps=(
            list[step],  # type: ignore[valid-type]
            Field(None, description=info["steps"].description),
        ),
        sinks=(
            list[sink],  # type: ignore[valid-type]
            Field(min_length=1, description=info["sinks"].description),
        ),
        providers=(
            providers_schema(registry),
            Field(None, description=info["providers"].description),
        ),
    )


def _steps(ctx: ResolveContext, value: object) -> list[PipelineStep] | None:
    """Resolve `steps:`; None if any step failed.

    Args:
        ctx: Collects errors and resolves references.
        value: The `steps:` value.

    Returns:
        The model steps and component steps, in order, or None after an error.
    """
    if value is None:
        return []
    if not isinstance(value, list):
        ctx.not_a_list(slots.STEPS, ("steps",))
        return None
    steps: list[PipelineStep] = []
    ok = True
    for i, item in enumerate(value):
        path: YamlPath = ("steps", i)
        if isinstance(item, dict) and "model" in item:
            extra = sorted(set(item) - {"model"})
            if extra:
                ctx.error(
                    (*path, extra[0]),
                    f"a model step takes no `{extra[0]}`",
                    "`{model: <model id>}`",
                    "move options into the model file",
                )
                ok = False
                continue
            model = ctx.model(item["model"], (*path, "model"))
            if model is None:
                ok = False
                continue
            steps.append(ModelStep(model))
            continue
        step = ctx.component(slots.STEPS, item, path)
        if step is None:
            ok = False
            continue
        steps.append(cast("Step", step))
    return steps if ok else None


__all__ = ["FOLDER", "Fields", "build_pipeline", "pipeline_schema"]
