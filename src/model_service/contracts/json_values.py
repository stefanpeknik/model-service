"""JSON values as the service stores and sends them: strict, finite, no NaN."""

from __future__ import annotations

from typing import Annotated

from pydantic import (
    BeforeValidator,
    ConfigDict,
    Field,
    TypeAdapter,
)
from pydantic import JsonValue as PydanticJsonValue


_JSON_VALUE: TypeAdapter[PydanticJsonValue] = TypeAdapter(
    PydanticJsonValue, config=ConfigDict(allow_inf_nan=False)
)


def _validate_json_value(value: object) -> PydanticJsonValue:
    """Validate a value as strict JSON, rejecting NaN and infinities.

    Args:
        value: Value to validate.

    Returns:
        The validated value.

    Raises:
        pydantic.ValidationError: The value is not strict JSON.
    """
    return _JSON_VALUE.validate_python(value, strict=True)


type JsonValue = Annotated[PydanticJsonValue, BeforeValidator(_validate_json_value)]


type NonEmptyString = Annotated[str, Field(strict=True, min_length=1)]


type JsonObject = Annotated[dict[str, JsonValue], Field(strict=True)]
