"""Validate the AppService response fields consumed by the gateway.

Unknown fields are ignored, but required fields, numeric bounds, and finite values are
checked before data reaches input preparation or image publication.
"""

from __future__ import annotations

import math
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from model_service.providers.empaia.app_service.gateway import EmpaiaGatewayError
from model_service.providers.empaia.app_service.types import Rectangle


type FiniteNumber = Annotated[float, Field(allow_inf_nan=False)]
type PositiveNumber = Annotated[FiniteNumber, Field(gt=0)]


class EmpaiaResponse(BaseModel):
    """Provide strict response validation and translation to a safe nonretryable gateway error."""

    model_config = ConfigDict(strict=True, extra="ignore", allow_inf_nan=False)

    @classmethod
    def parse(cls, value: object) -> Self:
        """Validate decoded provider data without exposing raw response content in errors.

        Args:
            value: The decoded JSON response.

        Returns:
            The validated response.

        Raises:
            EmpaiaGatewayError: Not retryable if the data fails validation.
        """
        try:
            return cls.model_validate(value)
        except ValidationError as error:
            raise EmpaiaGatewayError(
                "EMPAIA AppService returned an invalid response", retryable=False
            ) from error


class ConfigurationResponse(EmpaiaResponse):
    """Represent global and customer configuration sections returned by AppService."""

    global_values: dict[str, JsonValue] = Field(default_factory=dict, alias="global")
    customer_values: dict[str, JsonValue] = Field(
        default_factory=dict, alias="customer"
    )

    def merged(self) -> dict[str, JsonValue]:
        """Combine configuration sections only when their keys are disjoint, rejecting ambiguous values.

        Returns:
            The global and customer values in one mapping.

        Raises:
            EmpaiaGatewayError: Not retryable if both sections set a key.
        """
        if self.global_values.keys() & self.customer_values.keys():
            raise EmpaiaGatewayError(
                "EMPAIA configuration contains ambiguous values", retryable=False
            )
        return self.global_values | self.customer_values


class RectangleResponse(EmpaiaResponse):
    """Validate a finite two-coordinate origin and positive rectangle dimensions."""

    upper_left: Annotated[list[FiniteNumber], Field(min_length=2, max_length=2)]
    width: PositiveNumber
    height: PositiveNumber


def rectangle(value: object) -> Rectangle:
    """Floor rectangle origins and ceil dimensions when converting provider geometry to integer pixels.

    Args:
        value: The decoded rectangle input data.

    Returns:
        The rectangle in integer pixels.

    Raises:
        EmpaiaGatewayError: Not retryable if the data is not a valid rectangle.
    """
    parsed = RectangleResponse.parse(value)
    return Rectangle(
        x=math.floor(parsed.upper_left[0]),
        y=math.floor(parsed.upper_left[1]),
        width=math.ceil(parsed.width),
        height=math.ceil(parsed.height),
    )
