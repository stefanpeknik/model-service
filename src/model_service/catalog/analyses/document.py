"""Analysis files: resolving one with every load-time check, and their schema.

An analysis file names a source, steps (which may call served models) and
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

from components.base import slots
from model_service.catalog.analyses import checks
from model_service.catalog.analyses.declaration import Analysis
from model_service.catalog.analyses.exposure import exposure_schema, resolve_exposure
from model_service.catalog.analyses.model_step import ModelStep
from model_service.catalog.schema import usable


if TYPE_CHECKING:
    from components.base import Sink, Source, Step
    from model_service.catalog.analyses.declaration import AnalysisStep
    from model_service.catalog.context import ResolveContext
    from model_service.catalog.errors import YamlPath
    from model_service.registry import Registry


FOLDER = "analyses"
SUMMARY = "Analysis files: a source, steps (which may call served models) and sinks."


class Fields(BaseModel):
    """Every top-level key of an analysis file.

    The component keys are resolved through the registry; here they are only
    listed so a misspelt key names every key that fits.

    Attributes:
        summary: One line saying what the analysis does.
        source: The source.
        steps: Steps, in order.
        sinks: Sinks.
        expose: How the analysis is offered beyond the public API, e.g. in EMPAIA.
    """

    model_config = ConfigDict(extra="forbid")

    summary: str = Field("", description="One line saying what the analysis does.")
    source: Any = Field(None, description="A source: `{use: slide_tiles, …}`.")
    steps: Any = Field(
        None, description="Steps in order: `{model: <id>}` or `{use: <step>, …}`."
    )
    sinks: Any = Field(None, description="Sinks: `[{use: score_grid, output: …}, …]`.")
    expose: Any = Field(
        None, description="How the analysis is offered: `{empaia: {…}}` for an app."
    )


class ModelStepFields(BaseModel):
    """A model step: calls a served model on every tile."""

    model_config = ConfigDict(extra="forbid", title="model step")

    model: str = Field(
        description="A served-model ID, the name of a file in `served-models/`."
    )


def build_analysis(document: dict[str, Any], ctx: ResolveContext) -> Analysis | None:
    """Resolve the components of one analysis file and run every load-time check.

    Args:
        document: The analysis file after the environment's patch.
        ctx: Resolves components and served models and collects errors.

    Returns:
        The `Analysis`, or None if anything failed.
    """
    fields = ctx.fields(Fields, document)
    source = ctx.component(slots.SOURCE, document.get("source"), ("source",))
    steps = _steps(ctx, document.get("steps"))
    sinks = _list(ctx, slots.SINKS, document.get("sinks"), required=True)
    if fields is None or source is None or steps is None or ctx.errors:
        return None
    sink_list = cast("list[Sink]", sinks)
    filled = checks.fill_geometry(ctx, cast("Source", source), steps)
    columns = checks.fold_columns(ctx, filled, steps, sink_list)
    checks.check_settings(ctx, filled, steps, sink_list)
    checks.check_outputs(ctx, sink_list)
    if ctx.errors:
        return None
    analysis = Analysis(
        id=ctx.document_id,
        summary=fields.summary,
        source=filled,
        steps=tuple(steps),
        sinks=tuple(sink_list),
        inputs=filled.inputs(),
        columns=columns,
        declaration=document,
    )
    empaia = resolve_exposure(ctx, document.get("expose"), analysis)
    if ctx.errors:
        return None
    return replace(analysis, empaia=empaia)


def analysis_schema(registry: Registry) -> type[BaseModel]:
    """An analysis file, with the sources, steps and sinks there are.

    Args:
        registry: The components files may name.

    Returns:
        The file's model, for its JSON Schema.
    """
    step = usable(registry, slots.STEPS) | ModelStepFields
    sink = usable(registry, slots.SINKS)
    info = Fields.model_fields
    return create_model(
        "analysis",
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
        expose=(
            exposure_schema(registry),
            Field(None, description=info["expose"].description),
        ),
    )


def _steps(ctx: ResolveContext, value: object) -> list[AnalysisStep] | None:
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
        _not_a_list(ctx, slots.STEPS, ("steps",))
        return None
    steps: list[AnalysisStep] = []
    ok = True
    for i, item in enumerate(value):
        path: YamlPath = ("steps", i)
        if isinstance(item, dict) and "model" in item:
            extra = sorted(set(item) - {"model"})
            if extra:
                ctx.error(
                    (*path, extra[0]),
                    f"a model step takes no `{extra[0]}`",
                    "`{model: <served-model id>}`",
                    "move options into the served-model file",
                )
                ok = False
                continue
            model = ctx.served_model(item["model"], (*path, "model"))
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


def _list(
    ctx: ResolveContext, slot: slots.Slot, value: object, *, required: bool
) -> list[Any]:
    """Resolve a list of components under `slot`, reporting a missing one.

    Args:
        ctx: Collects errors.
        slot: The slot, `sinks:` for instance.
        value: Its value.
        required: Whether the list needs an entry.

    Returns:
        The components that resolved.
    """
    key = slot.key
    if value is None or (required and value == []):
        if required:
            ctx.error(
                (key,),
                f"missing `{key}:`",
                f"`{key}: [{{use: <name>, …}}, …]`",
                f"add at least one entry under `{key}:`",
            )
        return []
    if not isinstance(value, list):
        _not_a_list(ctx, slot, (key,))
        return []
    found = [ctx.component(slot, item, (key, i)) for i, item in enumerate(value)]
    return [c for c in found if c is not None]


def _not_a_list(ctx: ResolveContext, slot: slots.Slot, path: YamlPath) -> None:
    """Report a slot that holds something other than a list.

    Args:
        ctx: Collects errors.
        slot: The slot.
        path: Where it is in the file.
    """
    ctx.error(
        path,
        "not a list",
        f"`{slot.key}: [{{use: <name>, …}}, …]`",
        f"write `{slot.key}:` as a list",
    )


__all__ = ["FOLDER", "Fields", "analysis_schema", "build_analysis"]
