"""Wire the facade process: catalog, Ray Serve health check and API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import httpx

from model_service.builders.registry import BUILDERS, get_builder
from model_service.facade.api import create_app as create_api
from model_service.facade.catalog import ModelCatalog
from model_service.facade.health import DeploymentHealth
from model_service.facade.settings import SchedulerSettings


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi import FastAPI


def create_app(settings: SchedulerSettings | None = None) -> FastAPI:
    """Wire the production facade: catalog, Ray Serve health check and API.

    The shared HTTP client is closed when the app lifespan ends.

    Args:
        settings: Process settings; None reads them from the environment.

    Returns:
        The FastAPI application served by Uvicorn.

    Raises:
        CatalogError: If the model catalog file cannot be read or is invalid.
        UnknownBuilderError: If `BUILDER_IDS` names a builder that is not registered.
    """
    config = settings or SchedulerSettings.from_environment()
    catalog = ModelCatalog.from_file(config.model_catalog_path)
    builder_ids = config.builder_ids or frozenset(BUILDERS)
    for builder_id in builder_ids:
        get_builder(builder_id)
    client = httpx.AsyncClient()

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await client.aclose()

    return create_api(
        catalog=catalog,
        lifespan=lifespan,
        deployment_health=DeploymentHealth(
            client,
            origin=config.ray_serve_origin,
            routes=[f"/builders/{builder_id}" for builder_id in builder_ids]
            + [f"/{model.id}" for model in catalog],
        ),
    )
