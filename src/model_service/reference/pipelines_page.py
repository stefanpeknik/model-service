"""`docs/reference/pipelines.md`: every pipeline file of the catalog."""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_service.catalog.pipelines import FOLDER, ModelStep
from model_service.catalog.pipelines.document import Fields
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
    from model_service.catalog import Catalog
    from model_service.catalog.pipelines import Pipeline
    from model_service.catalog.pipelines.providers import EmpaiaPipelineSettings


def pipelines_page(catalog: Catalog) -> str:
    """Render the pipelines reference.

    Args:
        catalog: The base catalog.

    Returns:
        The page.
    """
    parts = [
        GENERATED,
        "# Pipelines",
        "Each file in `catalog/pipelines/` is one slide-in, results-out job. The "
        "file name is its ID: `POST /v1/pipelines/<id>/run` runs it and waits, "
        "`POST /v1/pipelines/<id>/schedule` runs it durably, and "
        "`GET /v1/pipelines` lists the deployed ones with their settings. To add "
        "one, see [Add a pipeline](../guides/add-a-pipeline.md).",
        "## Keys of a pipeline file",
        table(
            ["Key", "Description"],
            [
                [code(name), info.description or ""]
                for name, info in Fields.model_fields.items()
            ],
        ),
    ]
    for pipeline in catalog.pipelines.values():
        parts += _pipeline(pipeline)
    return "\n\n".join(p for p in parts if p) + "\n"


def _pipeline(pipeline: Pipeline) -> list[str]:
    """One pipeline's section.

    Args:
        pipeline: The pipeline.

    Returns:
        The section's Markdown blocks.
    """
    written = pipeline.declaration
    file = f"catalog/{FOLDER}/{pipeline.id}.yaml"
    rows = [
        ["Source", code(written["source"])],
        ["Tiles", _tiles(pipeline)],
        [
            "Steps",
            ", ".join(
                _step(s, w)
                for s, w in zip(pipeline.steps, written.get("steps") or [], strict=True)
            )
            or "none",
        ],
        ["Inputs", _inputs(pipeline)],
        ["Outputs", ", ".join(_output(s) for s in pipeline.sinks)],
        ["EMPAIA", _empaia(pipeline.empaia) if pipeline.empaia else "not offered"],
        ["File", f"[`{file}`](../../{file})"],
    ]
    schema = pipeline.settings_schema()
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
        f"## `{pipeline.id}`",
        pipeline.summary,
        facts(rows),
        "Per-run settings:" if settings else "",
        settings,
    ]


def _tiles(pipeline: Pipeline) -> str:
    """The tile geometry after the first model step filled it in.

    Args:
        pipeline: The resolved pipeline.

    Returns:
        `<size> px at <mpp> µm/px`, or a note that the source decides.
    """
    options = pipeline.source.options
    size, mpp = getattr(options, "tile_size", None), getattr(options, "mpp", None)
    if size is None or mpp is None:
        return "as the source reads them"
    return f"{size} px at {mpp} µm/px"


def _step(step: object, written: object) -> str:
    """A step as the pipeline's table lists it.

    Args:
        step: The resolved step.
        written: The step as the file wrote it.

    Returns:
        A link to a model step's model, or the step as written.
    """
    if isinstance(step, ModelStep):
        model_id = step.model.id
        return f"model [`{model_id}`](models.md#{model_id})"
    return code(written)


def _inputs(pipeline: Pipeline) -> str:
    """The pipeline's inputs and whether each is required.

    Args:
        pipeline: The resolved pipeline.

    Returns:
        The inputs, comma-separated.
    """
    return ", ".join(
        f"`{name}` ({'required' if required else 'optional'})"
        for name, required in pipeline.inputs.items()
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


def _empaia(app: EmpaiaPipelineSettings) -> str:
    """The EMPAIA app a pipeline becomes, in one line.

    Args:
        app: The resolved EMPAIA settings.

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
