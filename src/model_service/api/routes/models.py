"""`/v1/models`: the read-only model catalog.

Raw tiles go straight to the model's Ray Serve ingress (`POST /<model-id>/`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import APIRouter

from model_service.builders.registry import compatible_builders
from model_service.contracts.failures import UnknownModelError
from model_service.contracts.json_values import JsonValue
from model_service.contracts.model_declarations import (
    InferenceContract,
    ModelBinding,
)
from model_service.http_common.auth import Access, Action
from model_service.http_common.errors import ERRORS, lookup_error


if TYPE_CHECKING:
    from model_service.deployed_models.catalog import ModelCatalog


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
    return ModelView(
        id=model.id,
        contract=model.contract,
        parameters=model.model_dump(mode="json", exclude={"id", "contract"}),
        builders=compatible_builders(model),
    )


def create_router(catalog: ModelCatalog, access: Access) -> APIRouter:
    """Build the `/v1/models` catalog router.

    Args:
        catalog: Deployed models.
        access: Guards the routes.

    Returns:
        The router.
    """
    router = APIRouter(prefix="/v1/models")

    @router.get(
        "",
        dependencies=[access.require(Action.READ_CATALOG)],
    )
    async def list_models() -> list[ModelView]:
        return [model_view(model) for model in catalog]

    @router.get(
        "/{model_id}",
        responses=ERRORS,
        dependencies=[access.require(Action.READ_CATALOG)],
    )
    async def get_model(model_id: str) -> ModelView:
        try:
            return model_view(catalog.get(model_id))
        except UnknownModelError as error:
            raise lookup_error(error) from error

    return router
