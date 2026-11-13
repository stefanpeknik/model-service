"""The available builders, one declaration per builder package.

Adding a builder adds its package's `spec.SPEC` here. A provider that offers
builders as external apps may need an entry for it too.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_service.builders.heatmap.spec import SPEC as HEATMAP
from model_service.contracts.failures import UnknownBuilderError


if TYPE_CHECKING:
    from model_service.builders.declarations import BuilderSpec


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
