"""Validate the AppService response fields consumed by the gateway.

Unknown fields are ignored, but required fields, numeric bounds, and finite values are
checked before data reaches input preparation or image publication.
"""

from __future__ import annotations

import math
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from model_service.providers.empaia.app_service.gateway import EmpaiaGatewayError
from model_service.providers.empaia.app_service.types import (
    Rectangle,
    SlideLevel,
    SlideMetadata,
)


type NonEmptyString = Annotated[str, Field(min_length=1)]
type PositiveInteger = Annotated[int, Field(gt=0)]
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


def slide_metadata(value: object) -> SlideMetadata:
    """Convert validated slide input data into local geometry values.

    Args:
        value: The decoded slide input data.

    Returns:
        The slide's identity, size, pixel size and levels.

    Raises:
        EmpaiaGatewayError: Not retryable if the data is not a valid slide.
    """
    slide = SlideResponse.parse(value)
    return SlideMetadata(
        identifier=slide.id,
        width=slide.extent.x,
        height=slide.extent.y,
        pixel_size_nm_x=slide.pixel_size_nm.x,
        pixel_size_nm_y=slide.pixel_size_nm.y,
        levels=tuple(
            SlideLevel(index=index, width=level.extent.x, height=level.extent.y)
            for index, level in enumerate(slide.levels)
        ),
    )


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
