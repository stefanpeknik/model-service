"""Validate the AppService response fields consumed by the gateway.

Unknown fields are ignored, but required fields and finite values are checked
before the gateway uses the data.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from model_service.providers.empaia.gateway import EmpaiaGatewayError


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
