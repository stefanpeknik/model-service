"""Assemble the public HTTP API from the route modules.

`/healthz` answers while the process serves; `/healthz/deployments` checks Ray.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI, Response, status

from model_service.facade.health import (
    DeploymentHealthCheck,
    DeploymentState,
    DeploymentStatus,
)
from model_service.facade.routes import builders, models


if TYPE_CHECKING:
    from starlette.types import Lifespan

    from model_service.facade.catalog import ModelCatalog


def create_app(
    *,
    catalog: ModelCatalog,
    lifespan: Lifespan[FastAPI] | None = None,
    deployment_health: DeploymentHealthCheck | None = None,
) -> FastAPI:
    """Build the FastAPI app with health checks and the public routes.

    Args:
        catalog: Deployed models served under `/v1/models`.
        lifespan: Startup and shutdown context for the app.
        deployment_health: Checks Ray Serve routes; None reports `not_configured`.

    Returns:
        The assembled application.
    """
    app = FastAPI(title="model-service", lifespan=lifespan)

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

    app.include_router(models.create_router(catalog))
    app.include_router(builders.create_router())
    return app
