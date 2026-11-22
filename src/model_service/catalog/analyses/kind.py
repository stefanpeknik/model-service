"""The `analyses/` document kind."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from pydantic import BaseModel, ConfigDict, Field

from model_service.catalog.analyses import checks
from model_service.catalog.analyses.declaration import Analysis
from model_service.catalog.analyses.model_step import ModelStep
from model_service.catalog.kinds import DocumentKind
from model_service.catalog.served_models import FOLDER as SERVED_MODELS
from model_service.components import slots


if TYPE_CHECKING:
    from model_service.catalog.analyses.declaration import AnalysisStep
    from model_service.catalog.context import ResolveContext
    from model_service.catalog.errors import YamlPath
    from model_service.catalog.served_models import ServedModel
    from model_service.components import Sink, Source, Step


FOLDER = "analyses"


class Fields(BaseModel):
    """Every top-level key of an analysis file.

    The component keys are resolved through the registry; here they are only
    listed so a misspelt key names every key that fits.

    Attributes:
        summary: One line saying what the analysis does.
        source: The source.
        steps: Steps, in order.
        sinks: Sinks.
    """

    model_config = ConfigDict(extra="forbid")

    summary: str = Field("", description="One line saying what the analysis does.")
    source: Any = Field(None, description="A source: `{use: slide_tiles, …}`.")
    steps: Any = Field(
        None, description="Steps in order: `{model: <id>}` or `{use: <step>, …}`."
    )
    sinks: Any = Field(None, description="Sinks: `[{use: score_grid, output: …}, …]`.")


class Analyses(DocumentKind):
    """Analysis files: a source, steps (which may call served models) and sinks.

    ```yaml
    summary: Scores each tile of a prostate slide into a heatmap.
    source: {use: slide_tiles, stride_fraction: 0.5}
    steps:
      - {model: prostate-classifier-1}
    sinks:
      - {use: score_grid, output: heatmap}
    ```
    """

    folder = FOLDER
    title = "analysis"

    def resolve(self, document: dict[str, Any], ctx: ResolveContext) -> Any:
        """Resolve the components and run every load-time check.

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
        return Analysis(
            id=ctx.document_id,
            summary=fields.summary,
            source=filled,
            steps=tuple(steps),
            sinks=tuple(sink_list),
            inputs=filled.inputs(),
            columns=columns,
        )


def _steps(ctx: ResolveContext, value: object) -> list[AnalysisStep] | None:
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
            entry = ctx.reference(SERVED_MODELS, item["model"], (*path, "model"))
            if entry is None:
                ok = False
                continue
            steps.append(ModelStep(cast("ServedModel", entry.value)))
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
    ctx.error(
        path,
        "not a list",
        f"`{slot.key}: [{{use: <name>, …}}, …]`",
        f"write `{slot.key}:` as a list",
    )


__all__ = ["FOLDER", "Analyses", "Fields"]
