from __future__ import annotations

from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

from integrations.empaia.gateway import EmpaiaGatewayError


type NonEmptyString = Annotated[str, Field(min_length=1)]
type PositiveInteger = Annotated[int, Field(gt=0)]
type FiniteNumber = Annotated[float, Field(allow_inf_nan=False)]
type PositiveNumber = Annotated[FiniteNumber, Field(gt=0)]


class EmpaiaResponse(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore", allow_inf_nan=False)

    @classmethod
    def parse(cls, value: object) -> Self:
        try:
            return cls.model_validate(value)
        except ValidationError as error:
            raise EmpaiaGatewayError(
                "EMPAIA AppService returned an invalid response", retryable=False
            ) from error


class ConfigurationResponse(EmpaiaResponse):
    global_values: dict[str, JsonValue] = Field(default_factory=dict, alias="global")
    customer_values: dict[str, JsonValue] = Field(
        default_factory=dict, alias="customer"
    )

    def merged(self) -> dict[str, JsonValue]:
        if self.global_values.keys() & self.customer_values.keys():
            raise EmpaiaGatewayError(
                "EMPAIA configuration contains ambiguous values", retryable=False
            )
        return self.global_values | self.customer_values


class DirectAccessResponse(EmpaiaResponse):
    slide_id: NonEmptyString
    storage_path: NonEmptyString
    direct_access_supported: bool


class ExtentResponse(EmpaiaResponse):
    x: PositiveInteger
    y: PositiveInteger


class PixelSizeResponse(EmpaiaResponse):
    x: PositiveNumber
    y: PositiveNumber


class LevelResponse(EmpaiaResponse):
    extent: ExtentResponse


class SlideResponse(EmpaiaResponse):
    id: NonEmptyString
    extent: ExtentResponse
    pixel_size_nm: PixelSizeResponse
    levels: Annotated[list[LevelResponse], Field(min_length=1)]


class RectangleResponse(EmpaiaResponse):
    upper_left: Annotated[list[FiniteNumber], Field(min_length=2, max_length=2)]
    width: PositiveNumber
    height: PositiveNumber


class PixelmapResponse(EmpaiaResponse):
    id: NonEmptyString
