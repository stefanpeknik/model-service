"""How the EMPAIA converter or publisher of a declared type is picked.

The components live in `components/providers/empaia/`. An input's declared EAD type
picks its converter; an output's EAD type and its sink's data format pick its
publisher. The catalog picks them when a pipeline loads, and the EMPAIA
provider picks the same ones again from what admission recorded,
`{"type": <EAD type>, **options}` per input or output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from components import InputConverter, Publisher
from components._kinds.slots import EMPAIA_INPUTS, EMPAIA_OUTPUTS
from model_service.registry import default_registry


if TYPE_CHECKING:
    from collections.abc import Mapping

    from model_service.registry import Registry


TYPE = "type"


def converters_for(
    ead_type: str, registry: Registry | None = None
) -> list[type[InputConverter]]:
    """The converters of an EAD input type.

    Args:
        ead_type: The declared type, e.g. `rectangle`.
        registry: The components to look in; this repo's by default.

    Returns:
        The matching converter classes.
    """
    found = (registry or default_registry()).usable_in(EMPAIA_INPUTS)
    return [
        c for c in found if issubclass(c, InputConverter) and c.ead_type == ead_type
    ]


def publishers_for(
    format_name: str, ead_type: str, registry: Registry | None = None
) -> list[type[Publisher]]:
    """The publishers that read a data format and emit an EAD output type.

    Args:
        format_name: The sink's data format, e.g. `score_grid`.
        ead_type: The declared type, e.g. `continuous_pixelmap`.
        registry: The components to look in; this repo's by default.

    Returns:
        The matching publisher classes.
    """
    found = (registry or default_registry()).usable_in(EMPAIA_OUTPUTS)
    return [
        c
        for c in found
        if issubclass(c, Publisher)
        and c.reads_format == format_name
        and ead_type in c.emits
    ]


def converter(
    use: Mapping[str, Any], registry: Registry | None = None
) -> InputConverter:
    """Build the converter admission recorded.

    Args:
        use: `{"type": <EAD type>, **options}`.
        registry: The components to look in; this repo's by default.

    Returns:
        The converter.

    Raises:
        LookupError: If no converter, or more than one, takes the type.
        pydantic.ValidationError: If the options don't fit it.
    """
    options = {k: v for k, v in use.items() if k != TYPE}
    (found,) = _one(converters_for(str(use.get(TYPE)), registry), use)
    return cast("InputConverter", found.from_options(options))


def publisher(
    use: Mapping[str, Any], format_name: str, registry: Registry | None = None
) -> Publisher:
    """Build the publisher admission recorded, for a stored output's format.

    Args:
        use: `{"type": <EAD type>, **options}`.
        format_name: The stored output's data format.
        registry: The components to look in; this repo's by default.

    Returns:
        The publisher, with the type among its options.

    Raises:
        LookupError: If no publisher, or more than one, fits.
        pydantic.ValidationError: If the options don't fit it.
    """
    (found,) = _one(publishers_for(format_name, str(use.get(TYPE)), registry), use)
    return cast("Publisher", found.from_options(dict(use)))


def _one[C](found: list[C], use: Mapping[str, Any]) -> list[C]:
    """Exactly one match, or a lookup error naming how many there were.

    Args:
        found: The components matching `use`.
        use: The recorded EAD type and options.

    Returns:
        `found`, of length one.

    Raises:
        LookupError: If none or several match.
    """
    if len(found) != 1:
        raise LookupError(f"{len(found)} components take EMPAIA type {use.get(TYPE)!r}")
    return found
