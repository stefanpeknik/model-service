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
from model_service.contracts.failures import UnknownModelError
from model_service.contracts.json_values import JsonValue


if TYPE_CHECKING:
    from collections.abc import Callable

    from model_service.catalog import Catalog


@dataclass(frozen=True, slots=True)
class ModelView:
    """A served model, as the catalog lists it.

    Attributes:
        id: The model's id.
        declaration: Its served-model file after the environment's patch.
        digest: What decides its outputs, as the analysis runner checks it.
        analyses: IDs of the deployed analyses that call it.
    """

    id: str
    declaration: dict[str, JsonValue]
    digest: str
    analyses: list[str]


def model_view(model_id: str, catalog: Catalog) -> ModelView:
    """Describe a served model for the public catalog.

    Args:
        model_id: The served model's ID.
        catalog: The resolved served models and analyses.

    Returns:
        The model's ID, declaration, digest and the analyses calling it.

    Raises:
        UnknownModelError: If no deployed served model has this ID.
    """
    model = catalog.served_models.get(model_id)
    if model is None:
        raise UnknownModelError(model_id)
    return ModelView(
        id=model_id,
        declaration=model.declaration,
        digest=model.digest,
        analyses=[
            analysis.id
            for analysis in catalog.analyses.values()
            if model_id in analysis.models
        ],
    )


def create_router(catalog: Callable[[], Catalog], access: Access) -> APIRouter:
    """Build the `/v1/models` catalog router.

    Args:
        catalog: Gives the catalog in use.
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
        """List the deployed served models.

        Returns:
            Every served model in the catalog.
        """
        current = catalog()
        return [model_view(i, current) for i in current.served_models]

    @router.get(
        "/{model_id}",
        responses=ERRORS,
        dependencies=[access.require(Action.READ_CATALOG)],
    )
    async def get_model(model_id: str) -> ModelView:
        """Get one deployed served model.

        Args:
            model_id: The model's id.

        Returns:
            The model.

        Raises:
            HTTPException: 404 for an unknown model.
        """
        try:
            return model_view(model_id, catalog())
        except UnknownModelError as error:
            raise lookup_error(error) from error

    return router
