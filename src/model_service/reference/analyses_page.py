"""`docs/reference/analyses.md`: every analysis file of the catalog."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from model_service.catalog.analyses import FOLDER, ModelStep
from model_service.catalog.analyses.kind import Fields
from model_service.formats.known import get_format
from model_service.reference.markdown import (
    GENERATED,
    code,
    default,
    facts,
    schema_type,
    table,
)


if TYPE_CHECKING:
    from model_service.catalog import Catalog, Entry
    from model_service.catalog.analyses import Analysis
    from model_service.catalog.analyses.exposure import EmpaiaExposure


PAGE = Path("docs/reference/analyses.md")


def analyses_page(catalog: Catalog) -> str:
    """Render the analyses reference.

    Args:
        catalog: The base catalog.

    Returns:
        The page.
    """
    parts = [
        GENERATED,
        "# Analyses",
        "Each file in `catalog/analyses/` is one slide-in, results-out job. The "
        "file name is its ID: `POST /v1/analyses/<id>/run` runs it and waits, "
        "`POST /v1/analyses/<id>/schedule` runs it durably, and "
        "`GET /v1/analyses` lists the deployed ones with their settings. To add "
        "one, see [Add an analysis](../guides/add-an-analysis.md).",
        "## Keys of an analysis file",
        table(
            ["Key", "Description"],
            [
                [code(name), info.description or ""]
                for name, info in Fields.model_fields.items()
            ],
        ),
    ]
    for entry in catalog.entries(FOLDER).values():
        parts += _analysis(entry)
    return "\n\n".join(p for p in parts if p) + "\n"


def _analysis(entry: Entry) -> list[str]:
    """One analysis's section.

    Args:
        entry: The analysis's catalog entry.

    Returns:
        The section's Markdown blocks.
    """
    analysis: Analysis = entry.value
    written = entry.document
    file = entry.file
    rows = [
        ["Source", code(written["source"])],
        ["Tiles", _tiles(analysis)],
        [
            "Steps",
            ", ".join(
                _step(s, w)
                for s, w in zip(analysis.steps, written.get("steps") or [], strict=True)
            )
            or "none",
        ],
        ["Inputs", _inputs(analysis)],
        ["Outputs", ", ".join(_output(s) for s in analysis.sinks)],
        ["EMPAIA", _empaia(analysis.empaia) if analysis.empaia else "not exposed"],
        ["File", f"[`{file}`](../../{file})"],
    ]
    schema = analysis.settings_schema()
    defs = schema.get("$defs", {})
    settings = table(
        ["Setting", "Type", "Default", "Description"],
        [
            [
                code(name),
                schema_type(prop, defs),
                default(prop.get("default"), "default" not in prop),
                prop.get("description", ""),
            ]
            for name, prop in schema.get("properties", {}).items()
        ],
    )
    return [
        f"## `{analysis.id}`",
        analysis.summary,
        facts(rows),
        "Per-run settings:" if settings else "",
        settings,
    ]


def _tiles(analysis: Analysis) -> str:
    """The tile geometry after the first model step filled it in.

    Args:
        analysis: The resolved analysis.

    Returns:
        `<size> px at <mpp> µm/px`, or a note that the source decides.
    """
    options = analysis.source.options
    size, mpp = getattr(options, "tile_size", None), getattr(options, "mpp", None)
    if size is None or mpp is None:
        return "as the source reads them"
    return f"{size} px at {mpp} µm/px"


def _step(step: object, written: object) -> str:
    """A step as the analysis's table lists it.

    Args:
        step: The resolved step.
        written: The step as the file wrote it.

    Returns:
        A link to a model step's served model, or the step as written.
    """
    if isinstance(step, ModelStep):
        model_id = step.model.id
        return f"model [`{model_id}`](served-models.md#{model_id})"
    return code(written)


def _inputs(analysis: Analysis) -> str:
    """The analysis's inputs and whether each is required.

    Args:
        analysis: The resolved analysis.

    Returns:
        The inputs, comma-separated.
    """
    return ", ".join(
        f"`{name}` ({'required' if required else 'optional'})"
        for name, required in analysis.inputs.items()
    )


def _output(sink: object) -> str:
    """A sink's output with a link to its data format.

    Args:
        sink: The sink.

    Returns:
        `<output>: <format> (.<extension>)` in Markdown.
    """
    name = sink.options.output  # type: ignore[attr-defined]
    found = get_format(sink.format)  # type: ignore[attr-defined]
    return f"`{name}`: [`{found.name}`](formats.md#{found.name}) (`.{found.extension}`)"


def _empaia(app: EmpaiaExposure) -> str:
    """The EMPAIA app an analysis becomes, in one line.

    Args:
        app: The resolved exposure.

    Returns:
        Its name, modes, published outputs, job parameters and triggers.
    """
    modes = ", ".join(str(m) for m in app.modes)
    outputs = ", ".join(
        f"`{name}` as `{publisher.options.type}`"
        for name, publisher in app.outputs.items()
    )
    text = f"app “{app.name}” ({app.name_short}), modes: {modes}; publishes {outputs}"
    if app.parameters:
        text += f"; job parameters: {', '.join(map(code, app.parameters))}"
    if app.triggers:
        triggers = ", ".join(f"{t.tissue}/{t.stain}" for t in app.triggers)
        text += f"; preprocesses {triggers} slides"
    return text
