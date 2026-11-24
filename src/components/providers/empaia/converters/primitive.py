"""`primitive`: a standalone parameter, a setting the user fills in per job."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from components import EmpaiaInput, InputConverter
from model_service.providers.empaia.app_service.gateway import (
    EmpaiaGatewayError,
    PrimitiveType,
)


if TYPE_CHECKING:
    from pathlib import Path

    from model_service.contracts.json_values import JsonValue
    from model_service.providers.empaia.app_service.gateway import EmpaiaGateway


# The EMPAIA primitive of each JSON Schema type a setting may have.
PRIMITIVES = {
    "boolean": PrimitiveType.BOOL,
    "integer": PrimitiveType.INTEGER,
    "number": PrimitiveType.FLOAT,
    "string": PrimitiveType.STRING,
}


def primitive_of(setting: dict[str, Any]) -> PrimitiveType | None:
    """The EMPAIA primitive a setting is offered as.

    Args:
        setting: The setting's JSON Schema.

    Returns:
        The primitive, or None for a setting that is not one plain scalar.
    """
    return PRIMITIVES.get(setting.get("type", ""))


class Primitive(InputConverter):
    """An analysis setting offered as an EMPAIA primitive input in standalone mode.

    Its type and description come from the setting; the user's value is
    read when the job is admitted and overrides the file's.
    """

    name = "primitive"
    ead_type = "primitive"
    gives_file = False

    def io_fragment(self, item: EmpaiaInput) -> dict[str, Any]:
        """Declare the setting as a primitive of its type.

        Args:
            item: The input; `setting` is the setting's JSON Schema.

        Returns:
            The primitive entry with the setting's description.

        Raises:
            ValueError: If the setting is not one plain scalar.
        """
        primitive = primitive_of(item.setting)
        if primitive is None:
            raise ValueError(f"setting {item.name!r} is not a bool, number or string")
        return {"type": str(primitive), "description": item.setting["description"]}

    async def materialize(
        self, gateway: EmpaiaGateway, item: EmpaiaInput, directory: Path
    ) -> JsonValue:
        """Read the value the user gave.

        Args:
            gateway: The job's gateway.
            item: The input.
            directory: Unused.

        Returns:
            The primitive's value.

        Raises:
            EmpaiaGatewayError: Not retryable if the input is not a primitive.
        """
        value = await gateway.get_input_data(item.name)
        if not isinstance(value, dict) or value.get("type") not in PrimitiveType:
            raise EmpaiaGatewayError.invalid(
                f"EMPAIA input {item.name!r} is not a primitive"
            )
        return value.get("value")
