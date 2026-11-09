"""`/v1/models`: the model catalog."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import APIRouter

from model_service.builders.contracts import (
    InferenceContract,
    JsonValue,
    ModelBinding,
    UnknownModelError,
)
from model_service.facade.errors import ERRORS, ApiErrorCode, http_error


if TYPE_CHECKING:
    from model_service.facade.catalog import ModelCatalog


@dataclass(frozen=True, slots=True)
class ModelView:
    id: str
    contract: InferenceContract
    parameters: dict[str, JsonValue]
    builders: list[str]


def model_view(model: ModelBinding) -> ModelView:
    """Describe a model for the public catalog.

    Args:
        model: The model binding.

    Returns:
        The model's ID, contract, contract parameters and compatible builders.
    """
    from model_service.facade.catalog import ModelCatalog

    return ModelView(
        id=model.id,
        contract=model.contract,
        parameters=model.model_dump(mode="json", exclude={"id", "contract"}),
        builders=ModelCatalog.compatible_builders(model),
    )


def create_router(catalog: ModelCatalog) -> APIRouter:
    """Build the `/v1/models` router: the model catalog.

    Args:
        catalog: Deployed models.

    Returns:
        The router.
    """
    router = APIRouter(prefix="/v1/models")

    @router.get("")
    async def list_models() -> list[ModelView]:
        return [model_view(model) for model in catalog]

    @router.get("/{model_id}", responses=ERRORS)
    async def get_model(model_id: str) -> ModelView:
        try:
            return model_view(catalog.get(model_id))
        except UnknownModelError as error:
            raise http_error(
                404, ApiErrorCode.UNKNOWN_MODEL, "model is not deployed"
            ) from error

    return router
