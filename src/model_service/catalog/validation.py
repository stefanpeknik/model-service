"""Pydantic validation errors as catalog problems: what's wrong, what fits."""

from __future__ import annotations

import difflib
import typing
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from components.base.problems import Problem
from model_service.catalog.errors import render_path


if TYPE_CHECKING:
    from pydantic import ValidationError

    from model_service.catalog.errors import YamlPath


def problems(
    error: ValidationError, model: type[BaseModel], *, noun: str, owner: str
) -> list[tuple[YamlPath, Problem, str]]:
    """Explain each pydantic error of one validated mapping.

    Args:
        error: The validation error.
        model: The model that was validated.
        noun: What the keys are called, e.g. `option` or `key`.
        owner: Who has the keys, e.g. "`onnx`" or "a served model".

    Returns:
        Per error: its path inside the mapping, the problem, and the generic hint.
    """
    found = []
    for item in error.errors():
        loc, parent = _walk(model, tuple(item["loc"]))
        key = next((p for p in reversed(loc) if isinstance(p, str)), "")
        full = render_path(loc)
        detail: dict[str, Any] = {noun: full, "value": item.get("input")}
        if item["type"] == "extra_forbidden":
            fields = parent.model_fields
            close = difflib.get_close_matches(key, fields, n=1)
            whose = owner if len(loc) == 1 else render_path(loc[:-1])
            problem = Problem(
                "unknown_" + noun,
                f"unknown {noun} `{full}`",
                f"{noun.capitalize()}s of {whose}: {', '.join(sorted(fields))}",
                detail,
            )
            hint = f"did you mean `{close[0]}`?" if close else "remove it"
        elif item["type"] == "missing":
            problem = Problem(
                "missing_" + noun,
                f"missing {noun} `{full}`",
                _fits(parent, key),
                detail,
            )
            hint = f"add `{key}: …`"
        elif item["type"] == "union_tag_invalid":
            # A wrong `kind:` of a block with variants: point at the tag itself.
            tag_key = str(item["ctx"]["discriminator"]).strip("'")
            loc = (*loc, tag_key)
            detail["value"] = item["ctx"]["tag"]
            problem = Problem(
                "invalid_" + noun,
                f"`{item['ctx']['tag']}` is invalid: input should be "
                f"{item['ctx']['expected_tags'].replace(', ', ' or ')}",
                _fits(parent, key),
                detail,
            )
            hint = f"change `{tag_key}`"
        else:
            message = item["msg"][:1].lower() + item["msg"][1:]
            problem = Problem(
                "invalid_" + noun,
                f"`{item.get('input')}` is invalid: {message}",
                _fits(parent, key),
                detail,
            )
            hint = f"change `{key}`"
        found.append((loc, problem, hint))
    return found


def _walk(model: type[BaseModel], loc: YamlPath) -> tuple[YamlPath, type[BaseModel]]:
    """Follow an error location through nested blocks.

    Pydantic puts the tag of a tagged union (`scores` in `output.scores.range`)
    into the location; it is not in the YAML, so it is dropped.

    Args:
        model: The validated model.
        loc: The pydantic error location.

    Returns:
        The location as a YAML path, and the model owning its last key.
    """
    path: list[str | int] = []
    parent = model
    owner: type[BaseModel] | None = model
    parts = list(loc)
    while parts:
        part = parts.pop(0)
        path.append(part)
        if owner is None:
            continue
        parent = owner
        field = owner.model_fields.get(str(part))
        if field is None:
            owner = None
            continue
        annotation = _unalias(field.annotation)
        if typing.get_origin(annotation) is dict and parts:
            # A mapping of blocks by name: the next part is a name, not a key.
            path.append(parts.pop(0))
            annotation = _unalias(typing.get_args(annotation)[-1])
        variants = _models(annotation)
        tag = str(parts[0]) if parts else ""
        if len(variants) > 1 and tag in variants:
            parts.pop(0)
            owner = variants[tag]
        elif len(variants) == 1:
            (owner,) = variants.values()
        else:
            owner = None
    return tuple(path), parent


def _models(annotation: object) -> dict[str, type[BaseModel]]:
    """The block models an annotation may hold, by pydantic's union tag.

    Args:
        annotation: A field annotation.

    Returns:
        Tag (the discriminator's literal, else the class name) to model.
    """
    value = _unalias(annotation)
    if isinstance(value, type) and issubclass(value, BaseModel):
        return {_tag(value): value}
    if typing.get_origin(value) is dict:
        return {}
    found: dict[str, type[BaseModel]] = {}
    for arg in typing.get_args(value):
        found |= _models(arg)
    return found


def _unalias(annotation: object) -> object:
    """Unwrap a `type X = …` alias.

    Args:
        annotation: A type annotation.

    Returns:
        The aliased type, or the annotation itself.
    """
    return getattr(annotation, "__value__", annotation)


def _tag(model: type[BaseModel]) -> str:
    """The literal of a block's `kind` field, or its class name.

    Args:
        model: A block model.

    Returns:
        The tag pydantic uses for it in a tagged union.
    """
    kind = model.model_fields.get("kind")
    args = typing.get_args(kind.annotation) if kind is not None else ()
    return str(args[0]) if args else model.__name__


def _fits(model: type[BaseModel], key: str) -> str:
    """Describe what a key takes, from its field description or type.

    Args:
        model: The model the key belongs to.
        key: The key.

    Returns:
        `<key>: <description or type>`, or a pointer to the model's fields.
    """
    field = model.model_fields.get(key)
    if field is None:
        return f"See the fields of {model.__name__}"
    described = (field.description or "").strip().removesuffix(".")
    return f"{key}: {described or field.annotation}"
