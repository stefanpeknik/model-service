"""The available builders, one declaration per builder package.

Adding a builder adds its `declaration.SPEC` to `BUILDERS`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_service.builders.contracts import UnknownBuilderError


if TYPE_CHECKING:
    from model_service.builders.spec import BuilderSpec


BUILDERS: dict[str, BuilderSpec] = {}


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
