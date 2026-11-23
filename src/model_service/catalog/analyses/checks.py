"""The load-time checks of an analysis, each reporting through the context.

- geometry: the source takes `tile_size`/`mpp` from the first model step, and
  every model step must take the same tiles;
- the column fold over source, steps and sinks, with the models' dimensions;
- the frame fields sinks read;
- per-run settings, one flat name each;
- one sink per output name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from components.base import Problem, Source, per_run_fields, slots
from components.base.columns import resolved
from model_service.catalog.analyses.model_step import ModelStep


if TYPE_CHECKING:
    from collections.abc import Sequence

    from components.base import Col, Component, Sink
    from model_service.catalog.analyses.declaration import AnalysisStep
    from model_service.catalog.context import ResolveContext
    from model_service.catalog.errors import YamlPath
    from model_service.catalog.served_models import ServedModel


GEOMETRY = ("tile_size", "mpp")
MODEL_COLUMNS = {
    "scores": "a model step with `output: {kind: scores}`",
    "features": "a model step with `output: {kind: embedding, …}`",
}


def fill_geometry(
    ctx: ResolveContext, source: Source, steps: Sequence[AnalysisStep]
) -> Source:
    """Give the source the first model step's tile geometry where it has none.

    A conflicting value is reported and replaced by the model's, so the
    column fold doesn't report the same mistake again.

    Args:
        ctx: Collects errors.
        source: The source as the file gives it.
        steps: The resolved steps.

    Returns:
        The source with the first model step's `tile_size` and `mpp`; as
        given if it has no such options or there is no model step.
    """
    models = [(i, s.model) for i, s in enumerate(steps) if isinstance(s, ModelStep)]
    for i, other in models[1:]:
        first = models[0][1]
        if _geometry(other) != _geometry(first):
            ctx.error(
                ("steps", i, "model"),
                f"model step `{other.id}` takes {other.input.tile_size} px tiles at "
                f"{other.input.mpp} µm/px, but the first model step `{first.id}` "
                f"takes {first.input.tile_size} px at {first.input.mpp} µm/px",
                "Model steps that read the same tiles",
                "an analysis reads one tile geometry; split it into two analyses",
            )
    if not set(GEOMETRY) <= set(source.Options.model_fields):
        return source
    given = {k: getattr(source.options, k) for k in GEOMETRY}
    if not models:
        if None in given.values():
            problem = Problem(
                "missing_geometry",
                "a model-free analysis must set `tile_size` and `mpp`",
                f"E.g. `source: {{use: {source.name}, tile_size: 256, mpp: 1.0}}`",
                {"option": "tile_size"},
            )
            _report(ctx, ("source",), source, problem, "set `tile_size` and `mpp`")
        return source
    model = models[0][1]
    wanted = dict(zip(GEOMETRY, _geometry(model), strict=True))
    for key, value in given.items():
        if value is not None and value != wanted[key]:
            problem = Problem(
                "geometry_conflict",
                f"`{value}`, but model step `{model.id}` takes {wanted[key]}",
                "The first model step's value",
                {"option": key},
            )
            _report(ctx, ("source", key), source, problem, f"remove `{key}`")
    options = source.options.model_dump(exclude_unset=True) | wanted
    return cast("Source", source.from_options(options))


def fold_columns(
    ctx: ResolveContext,
    source: Source,
    steps: Sequence[AnalysisStep],
    sinks: Sequence[Sink],
) -> dict[str, Col]:
    """Check that every component reads columns an earlier one adds, in shape.

    Args:
        ctx: Collects errors and knows the component registry.
        source: The source, geometry filled in.
        steps: The resolved steps.
        sinks: The resolved sinks.

    Returns:
        The columns that reach the sinks.
    """
    available: dict[str, tuple[Col, str]] = {
        name: (col, f"`{source.name}`")
        for name, col in resolved(source.adds, source.options).items()
    }
    for i, step in enumerate(steps):
        if isinstance(step, ModelStep):
            reads, adds = step.columns()
            label = step.label
        else:
            reads = resolved(step.reads, step.options)
            adds = resolved(step.adds, step.options)
            label = f"`{step.name}`"
        _check_reads(ctx, ("steps", i), label, step, reads, available)
        available |= {name: (col, label) for name, col in adds.items()}
    for i, sink in enumerate(sinks):
        _check_frame(ctx, ("sinks", i), sink, source)
        reads = resolved(sink.reads, sink.options)
        _check_reads(ctx, ("sinks", i), f"`{sink.name}`", sink, reads, available)
    return {name: col for name, (col, _) in available.items()}


def check_settings(
    ctx: ResolveContext,
    source: Source,
    steps: Sequence[AnalysisStep],
    sinks: Sequence[Sink],
) -> None:
    """Refuse two components offering a per-run setting of the same name.

    Args:
        ctx: Collects errors.
        source: The source.
        steps: The resolved steps.
        sinks: The resolved sinks.
    """
    placed: list[tuple[YamlPath, Component]] = [(("source",), source)]
    placed += [
        (("steps", i), s) for i, s in enumerate(steps) if not isinstance(s, ModelStep)
    ]
    placed += [(("sinks", i), s) for i, s in enumerate(sinks)]
    owners: dict[str, tuple[YamlPath, Component]] = {}
    for path, component in placed:
        for name in sorted(per_run_fields(type(component.options))):
            if name in owners:
                first_path, first = owners[name]
                where = "".join(
                    f"[{p}]" if isinstance(p, int) else p for p in first_path
                )
                ctx.error(
                    path,
                    f"per-run setting `{name}` of `{component.name}` is also a "
                    f"setting of `{first.name}` at {where}",
                    "Settings are flat names, one per analysis",
                    "drop one of the two components or use another in its place",
                )
            else:
                owners[name] = (path, component)


def check_outputs(ctx: ResolveContext, sinks: Sequence[Sink]) -> None:
    """Refuse two sinks writing the same output.

    Args:
        ctx: Collects errors.
        sinks: The resolved sinks.
    """
    seen: dict[str, int] = {}
    for i, sink in enumerate(sinks):
        output = sink.options.output
        if output in seen:
            ctx.error(
                ("sinks", i, "output"),
                f"output `{output}` is also written by sinks[{seen[output]}]",
                "One sink per output name",
                "rename one `output:`",
            )
        seen.setdefault(output, i)


def _geometry(model: ServedModel) -> tuple[int, float]:
    """A served model's tile size and mpp.

    Args:
        model: The served model.

    Returns:
        `(tile_size, mpp)`.
    """
    return model.input.tile_size, model.input.mpp


def _check_reads(
    ctx: ResolveContext,
    path: YamlPath,
    label: str,
    reader: Any,
    reads: dict[str, Col],
    available: dict[str, tuple[Col, str]],
) -> None:
    """Report columns a component reads that are missing or don't fit.

    Args:
        ctx: Collects errors.
        path: Where the reader is in the file.
        label: The reader, as errors name it.
        reader: The component, whose `hint()` may fix the file.
        reads: The columns it reads.
        available: Each column so far, with the component that added it.
    """
    for name, want in reads.items():
        if name not in available:
            producers = [MODEL_COLUMNS[name]] if name in MODEL_COLUMNS else []
            producers += sorted(
                {
                    c.name
                    for slot in (slots.SOURCE, slots.STEPS)
                    for c in ctx.registry.usable_in(slot)
                    if name in c.adds
                }
            )
            problem = Problem(
                "missing_column",
                f"{label} reads `{name}`, which nothing before it adds",
                f"Added by: {', '.join(producers)}" if producers else "Nothing adds it",
                {"column": name},
            )
            _report(
                ctx,
                path,
                reader,
                problem,
                f"put a component that adds `{name}` before it",
            )
            continue
        have, producer = available[name]
        if not want.fits(have):
            problem = Problem(
                "column_mismatch",
                f"{label} reads `{name}` as {want}{_from(want)}, but {producer} "
                f"adds {have}{_from(have)}",
                f"A column {want}",
                {"column": name},
            )
            _report(
                ctx,
                path,
                reader,
                problem,
                f"pick a model or step whose `{name}` fits, "
                f"or a sink that reads {have}",
            )


def _check_frame(
    ctx: ResolveContext, path: YamlPath, sink: Sink, source: Source
) -> None:
    """Report frame fields a sink reads that the source doesn't give.

    Args:
        ctx: Collects errors.
        path: Where the sink is in the file.
        sink: The sink.
        source: The analysis's source.
    """
    missing = sorted(sink.frame_reads - source.frame.keys())
    if not missing:
        return
    givers = [
        c.name
        for c in ctx.registry.usable_in(slots.SOURCE)
        if issubclass(c, Source) and sink.frame_reads <= c.frame.keys()
    ]
    problem = Problem(
        "missing_frame_fields",
        f"`{sink.name}` reads frame fields {', '.join(missing)}, which "
        f"`{source.name}` doesn't give",
        f"Frame fields of `{source.name}`: {', '.join(sorted(source.frame)) or 'none'}",
        {"fields": missing},
    )
    example = f", e.g. {', '.join(givers)}" if givers else ""
    _report(ctx, path, sink, problem, f"use a source that gives them{example}")


def _from(col: Col) -> str:
    """Where a column's dimensions came from, in parentheses, if known.

    Args:
        col: The column.

    Returns:
        ` (<origin>)`, or an empty string.
    """
    return f" ({col.origin})" if col.origin else ""


def _report(
    ctx: ResolveContext, path: YamlPath, component: Any, problem: Problem, hint: str
) -> None:
    """Record a problem with the component's own hint, else `hint`.

    Args:
        ctx: Collects errors.
        path: Where the component is in the file.
        component: The component, whose `hint()` is asked first.
        problem: The failed check.
        hint: The fallback hint.
    """
    own = component.hint(problem) if hasattr(component, "hint") else None
    ctx.error(path, problem.message, problem.fits, own or hint)
