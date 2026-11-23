"""`docs/reference/components.md`: every registered component, by kind."""

from __future__ import annotations

import inspect
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from components.base import (
    KINDS,
    InputConverter,
    Publisher,
    Sink,
    Source,
    per_run_fields,
)
from model_service.catalog.analyses import FOLDER as ANALYSES
from model_service.catalog.analyses import ModelStep
from model_service.catalog.served_models import FOLDER as SERVED_MODELS
from model_service.reference.markdown import (
    GENERATED,
    code,
    docstring,
    facts,
    flow,
    options_table,
    table,
)
from model_service.registry import FOLDERS


if TYPE_CHECKING:
    from components.base import Col, Component, Slot
    from model_service.catalog import Catalog
    from model_service.registry import Registry


PAGE = Path("docs/reference/components.md")
SRC = Path(__file__).resolve().parents[2]


def components_page(registry: Registry, catalog: Catalog) -> str:
    """Render the components reference.

    Args:
        registry: The components.
        catalog: The base catalog, to list where each component is used.

    Returns:
        The page.
    """
    used = _used_in(catalog)
    parts = [
        GENERATED,
        "# Components",
        "A catalog file names a component with `{use: <name>, …options}`; "
        "EMPAIA converters and publishers are picked by `type:` instead. A "
        "component's kind decides where it may be named, and its folder under "
        "`src/components/` says the same. To write one, see "
        "[Write a component](../guides/write-a-component.md).",
        table(
            ["Kind", "Usable in", "Folder"],
            [
                [
                    f"[{kind.kind}](#{_plural(kind.kind)})",
                    _slots(kind.slots),
                    ", ".join(code(_folder(f)) for f in _folders_of(kind)),
                ]
                for kind in KINDS
            ],
        ),
    ]
    for kind in KINDS:
        members = [c for c in registry if c.kind == kind.kind]
        parts += [f"## {_plural(kind.kind).capitalize()}", docstring(kind)]
        if not members:
            parts.append("None yet.")
        parts += [_component(c, used[(c.kind, c.name)]) for c in members]
    return "\n\n".join(p for p in parts if p) + "\n"


def _component(component: type[Component], used: list[str]) -> str:
    """One component's section.

    Args:
        component: The component class.
        used: Links to the catalog files that name it.

    Returns:
        The section in Markdown.
    """
    built = component.from_options(component.example_options)
    rows = [
        ["Usable in", _slots(component.slots)],
        ["Example", code(_example(component))],
        *_kind_facts(built),
        ["Used in", ", ".join(used) or "no catalog file yet"],
        ["Code", _source_link(component)],
    ]
    options = options_table(component.Options, per_run_fields(component.Options))
    return "\n\n".join(
        p
        for p in (
            f"### `{component.name}` ({component.kind})",
            docstring(component),
            facts(rows),
            options,
        )
        if p
    )


def _kind_facts(built: Component) -> list[list[str]]:
    """The rows that depend on the kind: columns, formats, EAD types.

    Args:
        built: The component, built from its example options.

    Returns:
        Table rows of label and value.
    """
    reads, adds = type(built).reads, type(built).adds
    rows = []
    if isinstance(built, Source):
        rows.append(["Inputs (example options)", _inputs(built.inputs())])
        rows.append(["Frame", _columns(type(built).frame)])
    if reads:
        rows.append(["Reads", _columns(reads)])
    if adds:
        rows.append(["Adds", _columns(adds)])
    if isinstance(built, Sink):
        sink = type(built)
        rows.append(["Writes", f"[`{sink.format}`](formats.md#{sink.format})"])
        rows.append(
            ["Frame fields read", ", ".join(map(code, sorted(sink.frame_reads)))]
        )
        rows.append(["Rows in order", "yes" if sink.ordered else "no"])
    if isinstance(built, InputConverter):
        rows.append(["EAD input type", code(built.ead_type)])
        rows.append(["Gives", "a file" if built.gives_file else "a value"])
    if isinstance(built, Publisher):
        publisher = type(built)
        rows.append(
            [
                "Reads",
                f"[`{publisher.reads_format}`](formats.md#{publisher.reads_format})",
            ]
        )
        rows.append(["EAD output types", ", ".join(map(code, sorted(publisher.emits)))])
    return rows


def _example(component: type[Component]) -> str:
    """The component as a catalog file names it, with its example options.

    Args:
        component: The component class.

    Returns:
        The YAML flow mapping a file would write.
    """
    options: dict[str, Any] = dict(component.example_options)
    if issubclass(component, InputConverter):
        return flow({"inputs": {"<name>": {"type": component.ead_type, **options}}})
    if issubclass(component, Publisher):
        return flow({"outputs": {"<output>": options}})
    return flow({"use": component.name, **options})


def _columns(columns: dict[str, Col]) -> str:
    """Columns as their names and declared dtype and shape.

    Args:
        columns: Column name to its declaration.

    Returns:
        The columns, comma-separated.
    """
    return ", ".join(f"`{name}` {code(str(col))}" for name, col in columns.items())


def _inputs(inputs: dict[str, bool]) -> str:
    """A source's inputs and whether each is required.

    Args:
        inputs: Input name to whether it is required.

    Returns:
        The inputs, comma-separated.
    """
    return ", ".join(
        f"`{name}` ({'required' if required else 'optional'})"
        for name, required in inputs.items()
    )


def _slots(slots: tuple[Slot, ...]) -> str:
    """Where a component may be named.

    Args:
        slots: Its kind's slots.

    Returns:
        Each slot's document kind and key, comma-separated.
    """
    return ", ".join(f"{slot.document} `{slot.key}:`" for slot in slots)


def _folders_of(kind: type[Component]) -> list[str]:
    """The registry folders a kind's components live in.

    Args:
        kind: The kind's base class.

    Returns:
        Folder names in dotted form.
    """
    return [folder for folder, kinds in FOLDERS.items() if kind in kinds]


def _folder(folder: str) -> str:
    """A dotted folder name as a path from the repository root.

    Args:
        folder: E.g. `empaia.converters`.

    Returns:
        E.g. `src/components/empaia/converters/`.
    """
    return "src/components/" + folder.replace(".", "/") + "/"


def _plural(kind: str) -> str:
    """A kind's name in the plural, for headings.

    Args:
        kind: E.g. `loader`.

    Returns:
        E.g. `loaders`.
    """
    return f"{kind}s"


def _source_link(component: type[Component]) -> str:
    """A link to the file that defines a component.

    Args:
        component: The component class.

    Returns:
        The repository path as a Markdown link.
    """
    path = Path(inspect.getfile(component)).resolve().relative_to(SRC.parent)
    return f"[`{path}`](../../{path})"


def _used_in(catalog: Catalog) -> dict[tuple[str, str], list[str]]:
    """Links to the catalog files naming each component, by (kind, name).

    Args:
        catalog: The repository's catalog.

    Returns:
        Each component's links, in file order, without repeats.
    """
    used: dict[tuple[str, str], list[str]] = defaultdict(list)
    for folder in (SERVED_MODELS, ANALYSES):
        for entry in catalog.entries(folder).values():
            page = "served-models" if folder == SERVED_MODELS else "analyses"
            link = f"[`{entry.id}`]({page}.md#{entry.id})"
            for component in _components_of(entry.value):
                key = (component.kind, component.name)
                if link not in used[key]:
                    used[key].append(link)
    return used


def _components_of(value: Any) -> list[Component]:
    """Every component a resolved served model or analysis names.

    Args:
        value: A resolved served model or analysis.

    Returns:
        Its components, model steps left out.
    """
    if hasattr(value, "runtime"):
        return [value.weights, value.runtime, *value.pre, *value.post]
    steps = [s for s in value.steps if not isinstance(s, ModelStep)]
    exposed = []
    if value.empaia is not None:
        exposed = [*value.empaia.inputs.values(), *value.empaia.outputs.values()]
    return [value.source, *steps, *value.sinks, *exposed]
