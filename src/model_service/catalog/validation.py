"""Pydantic validation errors as catalog problems: what's wrong, what fits."""

from __future__ import annotations

import difflib
from typing import TYPE_CHECKING, Any

from components.base.problems import Problem


if TYPE_CHECKING:
    from pydantic import BaseModel, ValidationError

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
    fields = model.model_fields
    found = []
    for item in error.errors():
        loc: YamlPath = tuple(item["loc"])
        key = str(loc[0]) if loc else ""
        detail: dict[str, Any] = {noun: key, "value": item.get("input")}
        if item["type"] == "extra_forbidden":
            close = difflib.get_close_matches(key, fields, n=1)
            problem = Problem(
                "unknown_" + noun,
                f"unknown {noun} `{key}`",
                f"{noun.capitalize()}s of {owner}: {', '.join(sorted(fields))}",
                detail,
            )
            hint = f"did you mean `{close[0]}`?" if close else "remove it"
        elif item["type"] == "missing":
            problem = Problem(
                "missing_" + noun, f"missing {noun} `{key}`", _fits(model, key), detail
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
                _fits(model, key),
                detail,
            )
            hint = f"change `{tag_key}`"
        else:
            message = item["msg"][:1].lower() + item["msg"][1:]
            problem = Problem(
                "invalid_" + noun,
                f"`{item.get('input')}` is invalid: {message}",
                _fits(model, key),
                detail,
            )
            hint = f"change `{key}`"
        found.append((loc, problem, hint))
    return found


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
