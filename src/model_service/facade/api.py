"""Assemble the public HTTP API from the route modules.

`/healthz` answers while the process serves; `/healthz/deployments` checks Ray.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from model_service.facade.auth import Access
from model_service.facade.errors import ApiErrorCode
from model_service.facade.health import (
    DeploymentHealthCheck,
    DeploymentState,
    DeploymentStatus,
)
from model_service.facade.routes import builders, models


if TYPE_CHECKING:
    from starlette.types import Lifespan

    from model_service.facade.catalog import ModelCatalog
    from model_service.facade.inference import InferenceTarget
    from model_service.facade.ports import BuilderRunner


def create_app(
    *,
    catalog: ModelCatalog,
    builder_runner: BuilderRunner | None = None,
    inference_target: InferenceTarget | None = None,
    lifespan: Lifespan[FastAPI] | None = None,
    deployment_health: DeploymentHealthCheck | None = None,
    inference_max_request_bytes: int = 2 * 1024 * 1024,
    access: Access | None = None,
) -> FastAPI:
    """Build the FastAPI app with health checks and the public routes.

    Args:
        catalog: Deployed models served under `/v1/models`.
        builder_runner: Runs builders synchronously; None makes `/run` answer 503.
        inference_target: Runs raw model requests; None makes direct inference answer 503.
        lifespan: Startup and shutdown context for the app.
        deployment_health: Checks Ray Serve routes; None reports `not_configured`.
        inference_max_request_bytes: Largest raw inference body accepted, in bytes.
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
        # Do not echo the body: it may contain provider credentials.
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

    app.include_router(
        models.create_router(
            catalog, inference_target, inference_max_request_bytes, access
        )
    )
    app.include_router(builders.create_router(builder_runner, access))
    return app
