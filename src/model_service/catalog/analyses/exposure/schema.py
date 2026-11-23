"""The JSON schema of `expose:`, narrowed to the EAD types components handle."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model
from pydantic.fields import FieldInfo

from model_service.catalog.analyses.exposure.fields import EmpaiaFields, ExposeFields
from model_service.components import slots


if TYPE_CHECKING:
    from model_service.components.registry import Registry


def exposure_schema(registry: Registry) -> type[BaseModel]:
    """`expose:` for the file schema, with the EAD types components handle.

    Args:
        registry: The components files may name.

    Returns:
        The model of `expose:`.
    """
    input_types = sorted(
        {c.ead_type for c in registry.usable_in(slots.EMPAIA_INPUTS)}  # type: ignore[attr-defined]
    )
    output_types = sorted(
        {t for c in registry.usable_in(slots.EMPAIA_OUTPUTS) for t in c.emits}  # type: ignore[attr-defined]
    )

    def entry(name: str, types: list[str], what: str) -> type[BaseModel]:
        kind: Any = Literal[tuple(types)] if types else str
        return create_model(
            name,
            __config__=ConfigDict(extra="allow"),
            __doc__=f"`{{type: <EAD type>, …options}}`: {what}",
            type=(kind, Field(description="The EAD type.")),
        )

    inputs = entry("EmpaiaInput", input_types, "the type picks the converter.")
    outputs = entry(
        "EmpaiaOutput",
        output_types,
        "the type and the sink's data format pick the publisher.",
    )
    empaia = create_model(
        "EmpaiaApp",
        __base__=EmpaiaFields,
        __doc__="The EMPAIA app the analysis becomes.",
        inputs=(dict[str, inputs], _described("inputs")),  # type: ignore[valid-type]
        outputs=(dict[str, outputs], _described("outputs")),  # type: ignore[valid-type]
    )
    return create_model(
        "Expose",
        __base__=ExposeFields,
        __doc__="How the analysis is offered beyond the API.",
        empaia=(empaia | None, _described("empaia", ExposeFields)),
    )


def _described(name: str, model: type[BaseModel] = EmpaiaFields) -> FieldInfo:
    """A copy of `model`'s field `name`, for a schema that narrows its type."""
    return FieldInfo.merge_field_infos(model.model_fields[name])
