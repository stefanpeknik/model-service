"""Validate the AppService response fields consumed by the gateway.

Unknown fields are ignored, but required fields, numeric bounds, and finite values are
checked before data reaches workflow preparation or image publication.
"""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from integrations.empaia.gateway import EmpaiaGatewayError


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
            value: Decoded AppService response; unknown fields are ignored while
                consumed fields validate strictly.

        Returns:
            Self: New validated instance of the concrete AppService response model.

        Raises:
            EmpaiaGatewayError: Consumed AppService response fields fail strict schema
                validation.
        """
        try:
            return cls.model_validate(value)
        except ValidationError as error:
            raise EmpaiaGatewayError(
                "EMPAIA AppService returned an invalid response", retryable=False
            ) from error


class ConfigurationResponse(EmpaiaResponse):
    """Represent global and customer configuration sections returned by AppService.

    Attributes:
        global_values: Provider global configuration section.
        customer_values: Provider customer configuration section.
    """

    global_values: dict[str, JsonValue] = Field(default_factory=dict, alias="global")
    customer_values: dict[str, JsonValue] = Field(
        default_factory=dict, alias="customer"
    )

    def merged(self) -> dict[str, JsonValue]:
        """Combine configuration sections only when their keys are disjoint, rejecting ambiguous values.

        Returns:
            dict[str, JsonValue]: Combined global/customer configuration with disjoint
                keys.

        Raises:
            EmpaiaGatewayError: Global and customer sections contain an overlapping
                configuration key.
        """
        if self.global_values.keys() & self.customer_values.keys():
            raise EmpaiaGatewayError(
                "EMPAIA configuration contains ambiguous values", retryable=False
            )
        return self.global_values | self.customer_values


class DirectAccessResponse(EmpaiaResponse):
    """Describe a slide input's shared path and whether direct access is supported.

    Attributes:
        slide_id: External whole-slide input identifier.
        storage_path: Provider shared file path or authorized storage locator.
        direct_access_supported: Whether the provider permits direct shared-file access.
    """

    slide_id: NonEmptyString
    storage_path: NonEmptyString
    direct_access_supported: bool


class ExtentResponse(EmpaiaResponse):
    """Require positive integer image dimensions.

    Attributes:
        x: Positive image width in pixels.
        y: Positive image height in pixels.
    """

    x: PositiveInteger
    y: PositiveInteger


class PixelSizeResponse(EmpaiaResponse):
    """Require positive finite pixel sizes in nanometers.

    Attributes:
        x: Positive finite horizontal pixel size in nanometers.
        y: Positive finite vertical pixel size in nanometers.
    """

    x: PositiveNumber
    y: PositiveNumber


class LevelResponse(EmpaiaResponse):
    """Describe one pyramid level's image extent.

    Attributes:
        extent: Pixel width/height of this slide or pyramid level.
    """

    extent: ExtentResponse


class SlideResponse(EmpaiaResponse):
    """Validate slide identity, base geometry, physical pixel sizes, and at least one pyramid level.

    Attributes:
        id: External whole-slide image identifier.
        extent: Pixel width/height of this slide or pyramid level.
        pixel_size_nm: Physical pixel sizes in nanometers on both axes.
        levels: Available slide pyramid levels and their geometry.
    """

    id: NonEmptyString
    extent: ExtentResponse
    pixel_size_nm: PixelSizeResponse
    levels: Annotated[list[LevelResponse], Field(min_length=1)]


class RectangleResponse(EmpaiaResponse):
    """Validate a finite two-coordinate origin and positive rectangle dimensions.

    Attributes:
        upper_left: Two finite slide-space coordinates locating the rectangle origin.
        width: Pixel width or positive rectangle width.
        height: Pixel height or positive rectangle height.
    """

    upper_left: Annotated[list[FiniteNumber], Field(min_length=2, max_length=2)]
    width: PositiveNumber
    height: PositiveNumber


class PixelmapResponse(EmpaiaResponse):
    """Require the created pixelmap's nonempty identifier.

    Attributes:
        id: Identifier of the newly registered output pixelmap.
    """

    id: NonEmptyString
