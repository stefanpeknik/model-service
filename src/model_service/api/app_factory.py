"""Build the API process.

create_app assembles the FastAPI app from its routes; build_api_process wires
the catalog and dispatch into it.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import httpx
from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from model_service.api.health import (
    DeploymentHealth,
    DeploymentState,
    DeploymentStatus,
)
from model_service.api.routes import builders, models
from model_service.api.synchronous_runs import SynchronousBuilderService
from model_service.builders.registry import BUILDERS, get_builder
from model_service.deployed_models.catalog import ModelCatalog
from model_service.http_common.auth import Access
from model_service.http_common.errors import ApiErrorCode
from model_service.scheduler.builder_dispatch import HTTPBuilderDispatch
from model_service.scheduler.settings import SchedulerSettings


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from starlette.types import Lifespan


def create_app(
    *,
    catalog: ModelCatalog,
    builder_runner: SynchronousBuilderService | None = None,
    lifespan: Lifespan[FastAPI] | None = None,
    deployment_health: DeploymentHealth | None = None,
    access: Access | None = None,
) -> FastAPI:
    """Build the FastAPI app with health checks and the public routes.

    Args:
        catalog: Deployed models served under `/v1/models`.
        builder_runner: Runs builders synchronously; None makes `/run` answer 503.
        lifespan: Startup and shutdown context for the app.
        deployment_health: Checks Ray Serve routes; None reports `not_configured`.
        access: Guards the public API; the default lets everyone in.

    Returns:
        The assembled application.
    """
    access = access or Access()
    app = FastAPI(title="model-service", lifespan=lifespan)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(
        _request: Request, _error: RequestValidationError
    ) -> JSONResponse:
        # Do not echo the body: it may contain input access credentials.
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "detail": {
                    "code": ApiErrorCode.INVALID_REQUEST,
                    "message": "request body is invalid",
                    "retryable": False,
                }
            },
        )

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/healthz/deployments")
    async def deployment_status(response: Response) -> DeploymentStatus:
        if deployment_health is None:
            return DeploymentStatus(DeploymentState.NOT_CONFIGURED)
        result = await deployment_health.check()
        if result.status is not DeploymentState.READY:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return result

    app.include_router(models.create_router(catalog, access))
    app.include_router(builders.create_router(builder_runner, access))
    return app


def build_api_process(settings: SchedulerSettings | None = None) -> FastAPI:
    """Wire the production API process: catalog, builder dispatch and API.

    The shared HTTP client is closed when the app lifespan ends.

    Args:
        settings: Process settings; None reads them from the environment.

    Returns:
        The FastAPI application served by Uvicorn.

    Raises:
        CatalogError: If the model catalog file cannot be read or is invalid.
        UnknownBuilderError: If `BUILDER_IDS` names a builder that is not registered.
    """
    config = settings or SchedulerSettings()
    catalog = ModelCatalog.from_file(config.model_catalog_path)
    builder_ids = (
        frozenset(BUILDERS) if config.builder_ids is None else config.builder_ids
    )
    for builder_id in builder_ids:
        get_builder(builder_id)
    client = httpx.AsyncClient()
    dispatch = HTTPBuilderDispatch(client, origin=config.ray_serve_origin)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await client.aclose()

    return create_app(
        catalog=catalog,
        builder_runner=SynchronousBuilderService(catalog, dispatch),
        lifespan=lifespan,
        deployment_health=DeploymentHealth(
            client,
            origin=config.ray_serve_origin,
            routes=[f"/builders/{builder_id}" for builder_id in builder_ids]
            + [f"/{model.id}" for model in catalog],
        ),
    )
