"""JSON Schemas of catalog files, built from the component classes.

Each document kind gives a pydantic model of its file (`DocumentKind.schema`)
in which every component slot is a union of the components usable there, each
a `{use: <name>, …options}` object. Editors read the generated schemas for
autocompletion and inline errors; the catalog loader stays the authority, so
a file the schema accepts can still fail a load-time check.
"""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union, cast

from pydantic import ConfigDict, Field


if TYPE_CHECKING:
    from pydantic import BaseModel

    from model_service.components.component import Component
    from model_service.components.registry import Registry
    from model_service.components.slots import Slot


def with_use(component: type[Component]) -> type[BaseModel]:
    """A component's options plus the `use:` key that names it.

    Args:
        component: A component class.

    Returns:
        A model titled `<name> (<kind>)`, which forbids unknown keys as the
        component's `Options` do.
    """
    namespace = {
        "__module__": component.__module__,
        "__doc__": first_paragraph(component.__doc__),
        "__annotations__": {"use": Literal[component.name]},
        "model_config": ConfigDict(title=f"{component.name} ({component.kind})"),
        "use": Field(description=f"The {component.kind} `{component.name}`."),
    }
    name = f"{component.kind}_{component.name}"
    return cast("type[BaseModel]", type(name, (component.Options,), namespace))


def usable(registry: Registry, slot: Slot) -> Any:
    """The type of one `{use: …}` value in a slot: any component usable there.

    Args:
        registry: The components.
        slot: The slot.

    Returns:
        A type for a pydantic field: one model, a union told apart by `use`,
        or `Any` when no component fits the slot yet.
    """
    models = [with_use(c) for c in registry.usable_in(slot)]
    if not models:
        return Any
    if len(models) == 1:
        return models[0]
    return Annotated[Union[tuple(models)], Field(discriminator="use")]  # noqa: UP007


def file_schema(model: type[BaseModel]) -> dict[str, Any]:
    """The JSON Schema of a whole catalog file.

    Args:
        model: The document kind's file model.

    Returns:
        A draft 2020-12 schema.
    """
    schema = _without_attributes(model.model_json_schema())
    return {"$schema": "https://json-schema.org/draft/2020-12/schema", **schema}


def patch_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """The schema of an environment patch: the file's, with nothing required.

    A patch holds only what differs, so no key is required and a component's
    options may change without repeating its `use:`; unions become `anyOf`
    because, without `use:`, more than one component may fit.

    Args:
        schema: A file schema.

    Returns:
        The patch schema.
    """

    def relax(node: Any) -> Any:
        """Drop `required` and discriminators, turning `oneOf` into `anyOf`.

        Args:
            node: A part of the schema.

        Returns:
            The relaxed part.
        """
        if isinstance(node, dict):
            relaxed = {
                ("anyOf" if k == "oneOf" else k): relax(v)
                for k, v in node.items()
                if k not in {"required", "discriminator"}
            }
            if "title" in node and isinstance(node["title"], str) and node is schema:
                relaxed["title"] = f"{node['title']} (environment patch)"
            return relaxed
        if isinstance(node, list):
            return [relax(v) for v in node]
        return node

    return relax(copy.deepcopy(schema))


def _without_attributes(node: Any) -> Any:
    """Drop the `Attributes:` section models' docstrings bring into descriptions.

    Args:
        node: A part of a schema.

    Returns:
        The part, descriptions trimmed.
    """
    if isinstance(node, dict):
        return {
            k: v.split("\n\nAttributes:", 1)[0]
            if k == "description" and isinstance(v, str)
            else _without_attributes(v)
            for k, v in node.items()
        }
    if isinstance(node, list):
        return [_without_attributes(v) for v in node]
    return node


def first_paragraph(doc: str | None) -> str:
    """The first paragraph of a docstring, on one line.

    Args:
        doc: A docstring.

    Returns:
        Its first paragraph with line breaks joined; empty for no docstring.
    """
    return " ".join((doc or "").strip().split("\n\n", 1)[0].split())
