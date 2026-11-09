"""The available builders, one declaration per builder package.

Adding a builder adds its `declaration.SPEC` here; see
`docs/guides/adding-builders.md`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_service.builders.contracts import UnknownBuilderError
from model_service.builders.heatmap.declaration import SPEC as HEATMAP


if TYPE_CHECKING:
    from model_service.builders.contracts import BuilderOutputs, BuilderResult
    from model_service.builders.spec import BuilderSpec


BUILDERS: dict[str, BuilderSpec] = {spec.id: spec for spec in (HEATMAP,)}


def get_builder(builder_id: str) -> BuilderSpec:
    """Look up a builder declaration by id.

    Args:
        builder_id: Registered builder id.

    Returns:
        The builder's declaration.

    Raises:
        UnknownBuilderError: No builder has that id.
    """
    try:
        return BUILDERS[builder_id]
    except KeyError as error:
        raise UnknownBuilderError(builder_id) from error


def parse_result(builder_id: str, result: BuilderResult) -> BuilderOutputs:
    """Read a stored or received result as its builder's outputs model.

    Args:
        builder_id: The builder that produced the result.
        result: The generic result, as sent and stored.

    Returns:
        The result as the builder's `BuilderOutputs` subclass.

    Raises:
        UnknownBuilderError: No builder has this id.
        ValidationError: The result does not match the builder's current
            outputs, e.g. an output was renamed since it was stored.
    """
    return get_builder(builder_id).output_model.model_validate(result.outputs)
