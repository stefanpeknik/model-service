"""`/v1/models`: the read-only catalog of models.

Each model is listed with its declaration (the file after the
environment's patch), its digest and the pipelines that call it. Raw tiles go
straight to the model's Ray Serve ingress (`POST /<model-id>/`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from fastapi import APIRouter

from model_service.contracts.failures import UnknownModelError
from model_service.contracts.json_values import JsonValue
from model_service.control_plane.http.auth import Access, Action
from model_service.control_plane.http.errors import ERRORS, lookup_error


if TYPE_CHECKING:
    from collections.abc import Callable

    from model_service.catalog import Catalog


@dataclass(frozen=True, slots=True)
class ModelView:
    """A model, as the catalog lists it.

    Attributes:
        id: The model's id.
        declaration: Its model file after the environment's patch.
        digest: What decides its outputs, as the pipeline runner checks it.
        pipelines: IDs of the deployed pipelines that call it.
    """

    id: str
    declaration: dict[str, JsonValue]
    digest: str
    pipelines: list[str]


def model_view(model_id: str, catalog: Catalog) -> ModelView:
    """Describe a model for the public catalog.

    Args:
        model_id: The model's ID.
        catalog: The resolved models and pipelines.

    Returns:
        The model's ID, declaration, digest and the pipelines calling it.

    Raises:
        UnknownModelError: If no deployed model has this ID.
    """
    model = catalog.models.get(model_id)
    if model is None:
        raise UnknownModelError(model_id)
    return ModelView(
        id=model_id,
        declaration=model.declaration,
        digest=model.digest,
        pipelines=[
            pipeline.id
            for pipeline in catalog.pipelines.values()
            if model_id in pipeline.models
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
        """List the deployed models.

        Returns:
            Every model in the catalog.
        """
        current = catalog()
        return [model_view(i, current) for i in current.models]

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
            return model_view(model_id, catalog())
        except UnknownModelError as error:
            raise lookup_error(error) from error

    return router
