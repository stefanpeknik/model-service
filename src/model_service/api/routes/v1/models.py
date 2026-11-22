"""`/v1/models`: the read-only catalog of served models.

Each served model is listed with its declaration (the file after the
environment's patch), its digest and the analyses that call it. Raw tiles go
straight to the model's Ray Serve ingress (`POST /<model-id>/`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import APIRouter

from model_service.api.http.auth import Access, Action
from model_service.api.http.errors import ERRORS, lookup_error
from model_service.builders.registry import compatible_builders
from model_service.catalog.analyses import FOLDER as ANALYSES
from model_service.catalog.served_models import FOLDER as SERVED_MODELS
from model_service.contracts.failures import UnknownModelError
from model_service.contracts.json_values import JsonValue
from model_service.contracts.model_declarations import (
    InferenceContract,
    ModelBinding,
)


if TYPE_CHECKING:
    from model_service.api.model_catalog import ModelCatalogView
    from model_service.catalog import Catalog


@dataclass(frozen=True, slots=True)
class ModelView:
    """A model, as the catalog lists it.

    Attributes:
        id: The model's id.
        declaration: Its served-model file after the environment's patch;
            None for a model not yet migrated.
        digest: What decides its outputs, as the analysis runner checks it;
            None for a model not yet migrated.
        analyses: IDs of the deployed analyses that call it.
        contract: Its inference contract (builders, transitional).
        parameters: Its contract parameters, such as tile size and mpp.
        builders: Ids of the builders that accept it (transitional).
    """

    id: str
    declaration: dict[str, JsonValue] | None
    digest: str | None
    analyses: list[str]
    contract: InferenceContract
    parameters: dict[str, JsonValue]
    builders: list[str]


def model_view(model: ModelBinding, catalog: Catalog) -> ModelView:
    """Describe a model for the public catalog.

    Args:
        model: The model binding.
        catalog: The resolved served models and analyses.

    Returns:
        The model's ID, declaration, digest, the analyses calling it, its
        contract, contract parameters and compatible builders.
    """
    entry = catalog.entries(SERVED_MODELS).get(model.id)
    return ModelView(
        id=model.id,
        declaration=None if entry is None else entry.document,
        digest=None if entry is None else entry.value.digest,
        analyses=[
            analysis.id
            for analysis in catalog.entries(ANALYSES).values()
            if model.id in analysis.value.models
        ],
        contract=model.contract,
        parameters=model.model_dump(mode="json", exclude={"id", "contract"}),
        builders=compatible_builders(model),
    )


def create_router(catalog: ModelCatalogView, access: Access) -> APIRouter:
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
        """List the deployed models.

        Returns:
            Every model in the catalog.
        """
        return [model_view(model, catalog.resolved) for model in catalog]

    @router.get(
        "/{model_id}",
        responses=ERRORS,
        dependencies=[access.require(Action.READ_CATALOG)],
    )
    async def get_model(model_id: str) -> ModelView:
        """Get one deployed model.

        Args:
            model_id: The model's id.

        Returns:
            The model.

        Raises:
            HTTPException: 404 for an unknown model.
        """
        try:
            return model_view(catalog.get(model_id), catalog.resolved)
        except UnknownModelError as error:
            raise lookup_error(error) from error

    return router
