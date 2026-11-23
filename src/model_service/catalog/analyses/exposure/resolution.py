"""Resolve `expose.empaia` into converters and publishers, and run the EMPAIA checks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from pydantic import ValidationError

from model_service.catalog.analyses.exposure.fields import EmpaiaFields, ExposeFields
from model_service.catalog.validation import problems
from model_service.components import slots
from model_service.components.empaia import TYPE, converters_for, publishers_for
from model_service.components.empaia.input_converters.primitive import primitive_of
from model_service.providers.empaia.ead.vocabulary import AppMode, Trigger, ViewerConfig


if TYPE_CHECKING:
    from collections.abc import Mapping

    from model_service.catalog.analyses.declaration import Analysis
    from model_service.catalog.context import ResolveContext
    from model_service.catalog.errors import YamlPath
    from model_service.components import InputConverter, Publisher


WSI = "wsi"
# The input types an analysis need not declare, by input name.
DEFAULT_INPUT_TYPES = {"slide": WSI, "region_of_interest": "rectangle"}
PATH: YamlPath = ("expose", "empaia")


@dataclass(frozen=True)
class EmpaiaExposure:
    """The EMPAIA app of an analysis, resolved and checked.

    Attributes:
        name: The app's name.
        name_short: Its short name.
        description: What it finds, without the modes' sentences.
        modes: The offered modes, standalone first.
        parameters: The settings users fill in per standalone job.
        triggers: The slides EMPAIA preprocesses with it.
        inputs: The converter of each input, in the analysis's input order.
        slide: The input that is the slide, which other entries reference.
        outputs: The publisher of each published output, in sink order.
        viewer: The case browser's display settings.
    """

    name: str
    name_short: str
    description: str
    modes: tuple[AppMode, ...]
    parameters: tuple[str, ...]
    triggers: tuple[Trigger, ...]
    inputs: Mapping[str, InputConverter]
    slide: str
    outputs: Mapping[str, Publisher]
    viewer: ViewerConfig = field(default_factory=ViewerConfig)

    def mode_inputs(self, mode: AppMode, required: Mapping[str, bool]) -> list[str]:
        """The inputs a mode takes: every one standalone, the required ones in preprocessing.

        Args:
            mode: The mode.
            required: Whether each input is required, the analysis's `inputs`.

        Returns:
            Input names, in the analysis's order.
        """
        return [
            name
            for name in self.inputs
            if mode is AppMode.STANDALONE or required.get(name, False)
        ]


def resolve_exposure(
    ctx: ResolveContext, value: object, analysis: Analysis
) -> EmpaiaExposure | None:
    """Resolve `expose:` and run the EMPAIA checks.

    Args:
        ctx: Collects errors.
        value: The `expose:` value; None when the analysis has none.
        analysis: The resolved analysis.

    Returns:
        The EMPAIA app, or None without `expose.empaia` or after an error.
    """
    if value is None:
        return None
    expose = ctx.fields(ExposeFields, value, ("expose",))
    if expose is None or expose.empaia is None:
        return None
    fields = ctx.fields(EmpaiaFields, expose.empaia, PATH)
    if fields is None:
        return None
    inputs = _inputs(ctx, fields, analysis)
    slide = _slide(ctx, inputs)
    parameters = _parameters(ctx, fields, analysis)
    outputs = _outputs(ctx, fields, analysis)
    modes = fields.modes
    if (
        slide is not None
        and modes.preprocessing is not None
        and (needed := [n for n, r in analysis.inputs.items() if r and n != slide])
    ):
        ctx.error(
            (*PATH, "modes", "preprocessing"),
            f"preprocessing starts a job with the slide alone, but {analysis.id} "
            f"requires {', '.join(needed)}",
            "Standalone mode only, or an optional input",
            "drop `preprocessing:` or make the input optional in the source",
        )
    if inputs is None or slide is None or parameters is None or outputs is None:
        return None
    if ctx.errors:
        return None
    return EmpaiaExposure(
        name=fields.name,
        name_short=fields.name_short,
        description=fields.description,
        modes=tuple(
            mode
            for mode, settings in (
                (AppMode.STANDALONE, modes.standalone),
                (AppMode.PREPROCESSING, modes.preprocessing),
            )
            if settings is not None
        ),
        parameters=parameters,
        triggers=modes.preprocessing.triggers if modes.preprocessing else (),
        inputs=inputs,
        slide=slide,
        outputs=outputs,
        viewer=fields.viewer,
    )


def _inputs(
    ctx: ResolveContext, fields: EmpaiaFields, analysis: Analysis
) -> dict[str, InputConverter] | None:
    """Pick each input's converter by its declared or default type.

    Args:
        ctx: Collects errors.
        fields: The checked `expose.empaia` keys.
        analysis: The resolved analysis.

    Returns:
        The converter by input name, or None after an error.
    """
    ok = True
    for name in sorted(set(fields.inputs) - set(analysis.inputs)):
        ctx.error(
            (*PATH, "inputs", name),
            f"{analysis.id} has no input `{name}`",
            f"Inputs: {', '.join(analysis.inputs)}",
            "list only inputs the source reads",
        )
        ok = False
    converters: dict[str, InputConverter] = {}
    for name in analysis.inputs:
        declared = dict(fields.inputs.get(name, {}))
        ead_type = declared.pop(TYPE, DEFAULT_INPUT_TYPES.get(name))
        path: YamlPath = (*PATH, "inputs", name)
        if ead_type is None:
            ctx.error(
                path,
                f"input `{name}` has no EMPAIA type",
                f"`{name}: {{type: <EAD type>}}`",
                "declare the input's EAD type, e.g. `wsi` or `rectangle`",
            )
            ok = False
            continue
        found = converters_for(str(ead_type), ctx.registry)
        if len(found) != 1:
            ctx.error(
                (*path, TYPE),
                f"{len(found)} converters take EMPAIA type `{ead_type}`",
                f"Input types: {', '.join(_input_types(ctx))}",
                "pick a type one converter takes",
            )
            ok = False
            continue
        built = _build(ctx, found[0], declared, path)
        if built is None:
            ok = False
            continue
        converters[name] = cast("InputConverter", built)
    return converters if ok else None


def _slide(
    ctx: ResolveContext, inputs: Mapping[str, InputConverter] | None
) -> str | None:
    """The one input that is the slide.

    Args:
        ctx: Collects errors.
        inputs: The converter by input name, or None if they failed.

    Returns:
        The slide's input name, or None after an error.
    """
    if inputs is None:
        return None
    slides = [n for n, c in inputs.items() if c.ead_type == WSI]
    if len(slides) == 1:
        return slides[0]
    ctx.error(
        (*PATH, "inputs"),
        f"an EMPAIA app needs exactly one `wsi` input, found {len(slides)}",
        "One input of type `wsi`, the slide",
        "declare the slide input as `{type: wsi}` and no other",
    )
    return None


def _parameters(
    ctx: ResolveContext, fields: EmpaiaFields, analysis: Analysis
) -> tuple[str, ...] | None:
    """Check the standalone parameters are plain per-run settings.

    Args:
        ctx: Collects errors.
        fields: The checked `expose.empaia` keys.
        analysis: The resolved analysis.

    Returns:
        The parameters, empty without standalone mode, or None after an error.
    """
    standalone = fields.modes.standalone
    if standalone is None:
        return ()
    schema = analysis.settings_schema()["properties"]
    ok = True
    names = standalone.parameters
    for i, name in enumerate(names):
        path: YamlPath = (*PATH, "modes", "standalone", "parameters", i)
        if name not in schema:
            ctx.error(
                path,
                f"`{name}` is not a setting of {analysis.id}",
                f"Settings: {', '.join(schema) or 'none'}",
                "list per-run settings (`PerRun` options) only",
            )
            ok = False
        elif primitive_of(schema[name]) is None:
            ctx.error(
                path,
                f"setting `{name}` is not a bool, number or string",
                "A setting of one plain scalar type",
                "leave it out of the parameters",
            )
            ok = False
        elif name in analysis.inputs or names.index(name) != i:
            ctx.error(
                path,
                f"`{name}` is listed twice or is also an input",
                "Each setting once",
                "remove the repeat",
            )
            ok = False
    return tuple(names) if ok else None


def _outputs(
    ctx: ResolveContext, fields: EmpaiaFields, analysis: Analysis
) -> dict[str, Publisher] | None:
    """Pick each listed output's publisher by its sink's format and declared type.

    Args:
        ctx: Collects errors.
        fields: The checked `expose.empaia` keys.
        analysis: The resolved analysis.

    Returns:
        The publisher by output name, or None after an error.
    """
    sinks = {s.options.output: s for s in analysis.sinks}
    publishers: dict[str, Publisher] = {}
    ok = True
    for name, declared in fields.outputs.items():
        path: YamlPath = (*PATH, "outputs", name)
        sink = sinks.get(name)
        if sink is None:
            ctx.error(
                path,
                f"{analysis.id} has no output `{name}`",
                f"Outputs: {', '.join(sinks)}",
                "list sink outputs only",
            )
            ok = False
            continue
        ead_type = declared.get(TYPE)
        found = publishers_for(sink.format, str(ead_type), ctx.registry)
        if len(found) != 1:
            fits = sorted(
                t
                for c in ctx.registry.usable_in(slots.EMPAIA_OUTPUTS)
                if getattr(c, "reads_format", None) == sink.format
                for t in getattr(c, "emits", ())
            )
            ctx.error(
                (*path, TYPE),
                f"no publisher takes `{sink.format}` as EMPAIA type `{ead_type}`",
                f"Types for `{sink.format}`: {', '.join(fits) or 'none yet'}",
                "pick one of them, or leave the output out of EMPAIA",
            )
            ok = False
            continue
        built = _build(ctx, found[0], declared, path)
        if built is None:
            ok = False
            continue
        publisher = cast("Publisher", built)
        if (problem := publisher.check(analysis.columns)) is not None:
            ctx.error(
                (*path, TYPE),
                problem.message,
                problem.fits,
                publisher.hint(problem) or "change the model or the output's type",
            )
            ok = False
            continue
        publishers[name] = publisher
    return publishers if ok else None


def _input_types(ctx: ResolveContext) -> list[str]:
    """The EAD input types the registry's converters take.

    Args:
        ctx: Holds the registry.

    Returns:
        The types, sorted.
    """
    return sorted(
        {
            getattr(c, "ead_type", "")
            for c in ctx.registry.usable_in(slots.EMPAIA_INPUTS)
        }
    )


def _build(
    ctx: ResolveContext, component: Any, options: dict[str, Any], path: YamlPath
) -> Any:
    """Build a converter or publisher, reporting options that don't fit.

    Args:
        ctx: Collects errors.
        component: The converter or publisher class.
        options: Its options from the file, `type` aside for converters.
        path: Where the entry is in the file.

    Returns:
        The component, or None after an error.
    """
    try:
        return component.from_options(options)
    except ValidationError as error:
        for loc, problem, hint in problems(
            error, component.Options, noun="option", owner=f"`{component.name}`"
        ):
            ctx.error((*path, *loc), problem.message, problem.fits, hint)
        return None
