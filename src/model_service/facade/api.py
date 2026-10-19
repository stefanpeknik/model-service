"""Assemble the public HTTP API from the route modules.

Readiness checks durable storage; `/healthz/deployments` checks Ray.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from model_service.facade.health import DeploymentState, DeploymentStatus
from model_service.facade.routes import builders, empaia, executions, models
from model_service.facade.routes.common import ApiErrorCode, http_error
from model_service.scheduler.contracts import SchedulingUnavailableError


if TYPE_CHECKING:
    from starlette.types import Lifespan

    from model_service.facade.catalog import ModelCatalog
    from model_service.facade.routes.builders import BuilderRunner
    from model_service.facade.routes.common import Scheduler
    from model_service.scheduler.inference import InferenceTarget


class DeploymentHealth(Protocol):
    async def check(self) -> DeploymentStatus: ...


def create_app(
    scheduler: Scheduler,
    *,
    catalog: ModelCatalog,
    builder_runner: BuilderRunner | None = None,
    inference_target: InferenceTarget | None = None,
    lifespan: Lifespan[FastAPI] | None = None,
    deployment_health: DeploymentHealth | None = None,
    inference_max_request_bytes: int = 2 * 1024 * 1024,
    empaia_enabled: bool = False,
) -> FastAPI:
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

    @app.get("/readyz")
    async def readiness() -> dict[str, str]:
        try:
            await scheduler.check_ready()
        except SchedulingUnavailableError as error:
            raise http_error(
                503,
                ApiErrorCode.SCHEDULING_UNAVAILABLE,
                "execution repository is temporarily unavailable",
                retryable=True,
            ) from error
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
            scheduler, catalog, inference_target, inference_max_request_bytes
        )
    )
    app.include_router(builders.create_router(scheduler, builder_runner))
    if empaia_enabled:
        app.include_router(empaia.create_router(scheduler))
    app.include_router(executions.create_router(scheduler))
    return app
