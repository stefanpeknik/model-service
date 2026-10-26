"""Validate the AppService response fields consumed by the gateway.

Unknown fields are ignored, but required fields, numeric bounds, and finite values are
checked before data reaches input preparation or image publication.
"""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from model_service.providers.empaia.gateway import EmpaiaGatewayError


type NonEmptyString = Annotated[str, Field(min_length=1)]
type PositiveInteger = Annotated[int, Field(gt=0)]
type FiniteNumber = Annotated[float, Field(allow_inf_nan=False)]
type PositiveNumber = Annotated[FiniteNumber, Field(gt=0)]


class EmpaiaResponse(BaseModel):
    """Provide strict response validation and translation to a safe nonretryable gateway error."""

    model_config = ConfigDict(strict=True, extra="ignore", allow_inf_nan=False)

    @classmethod
    def parse(cls, value: object) -> Self:
        """Validate decoded provider data without exposing raw response content in errors."""
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
        """Combine configuration sections only when their keys are disjoint, rejecting ambiguous values."""
        if self.global_values.keys() & self.customer_values.keys():
            raise EmpaiaGatewayError(
                "EMPAIA configuration contains ambiguous values", retryable=False
            )
        return self.global_values | self.customer_values


class DirectAccessResponse(EmpaiaResponse):
    """Describe a slide input's shared path and whether direct access is supported."""

    slide_id: NonEmptyString
    storage_path: NonEmptyString
    direct_access_supported: bool


class ExtentResponse(EmpaiaResponse):
    """Require positive integer image dimensions."""

    x: PositiveInteger
    y: PositiveInteger


class PixelSizeResponse(EmpaiaResponse):
    """Require positive finite pixel sizes in nanometers."""

    x: PositiveNumber
    y: PositiveNumber


class LevelResponse(EmpaiaResponse):
    """Describe one pyramid level's image extent."""

    extent: ExtentResponse


class SlideResponse(EmpaiaResponse):
    """Validate slide identity, base geometry, physical pixel sizes, and at least one pyramid level."""

    id: NonEmptyString
    extent: ExtentResponse
    pixel_size_nm: PixelSizeResponse
    levels: Annotated[list[LevelResponse], Field(min_length=1)]


class RectangleResponse(EmpaiaResponse):
    """Validate a finite two-coordinate origin and positive rectangle dimensions."""

    upper_left: Annotated[list[FiniteNumber], Field(min_length=2, max_length=2)]
    width: PositiveNumber
    height: PositiveNumber


class PixelmapResponse(EmpaiaResponse):
    """Require the created pixelmap's nonempty identifier."""

    id: NonEmptyString
