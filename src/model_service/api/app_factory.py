"""Build the API process.

create_app assembles the FastAPI app from its routes; build_api_process wires
the catalog, scheduler, providers and dispatch into it.
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
from model_service.api.routes import builders, executions, models
from model_service.api.synchronous_runs import SynchronousBuilderService
from model_service.builders.registry import BUILDERS, get_builder
from model_service.deployed_models.catalog import ModelCatalog
from model_service.http_common.auth import Access
from model_service.http_common.errors import ApiErrorCode, http_error
from model_service.providers.registry import PROVIDERS
from model_service.scheduler.builder_dispatch import HTTPBuilderDispatch
from model_service.scheduler.execution_scheduler import ExecutionScheduler
from model_service.scheduler.internal_state import RetryPolicy
from model_service.scheduler.public import SchedulingUnavailableError
from model_service.scheduler.settings import SchedulerSettings
from model_service.scheduler.store.postgres import PostgresExecutionRepository
from model_service.scheduler.store.protected_context import ProtectedContextCipher


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from starlette.types import Lifespan

    from model_service.contracts.model_declarations import ModelBinding
    from model_service.providers.provider import RouteFactory
    from model_service.scheduler.protocols import Scheduler
    from model_service.scheduler.public import ModelResolver


def model_resolver(catalog: ModelCatalog) -> ModelResolver:
    """Make the scheduler's resolver from builder and model IDs to a validated model.

    Args:
        catalog: Deployed models to look up.

    Returns:
        A function that returns the model binding, raising `UnknownBuilderError`,
        `UnknownModelError` or `IncompatibleModelError` when the pair is unusable.
    """

    def resolve(builder_id: str, model_id: str) -> ModelBinding:
        spec = get_builder(builder_id)
        model = catalog.get(model_id)
        spec.check_model(model)
        return model

    return resolve


def create_app(
    scheduler: Scheduler,
    *,
    catalog: ModelCatalog,
    builder_runner: SynchronousBuilderService | None = None,
    lifespan: Lifespan[FastAPI] | None = None,
    deployment_health: DeploymentHealth | None = None,
    access: Access | None = None,
    provider_routes: Sequence[RouteFactory] = (),
) -> FastAPI:
    """Build the FastAPI app with health checks, public routes and provider routes.

    Args:
        scheduler: Admits, reads and cancels durable executions.
        catalog: Deployed models served under `/v1/models`.
        builder_runner: Runs builders synchronously; None makes `/run` answer 503.
        lifespan: Startup and shutdown context for the app.
        deployment_health: Checks Ray Serve routes; None reports `not_configured`.
        access: Guards the public API; the default lets everyone in. Providers
            guard their own routes.
        provider_routes: Router factories contributed by providers.

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
    async def health() -> JSONResponse:
        # Liveness: an API process whose worker stopped would hold jobs forever.
        if not scheduler.alive:
            return JSONResponse({"status": "worker stopped"}, status_code=503)
        return JSONResponse({"status": "ok"})

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

    app.include_router(models.create_router(catalog, access))
    app.include_router(builders.create_router(scheduler, builder_runner, access))
    for routes in provider_routes:
        app.include_router(routes(scheduler))
    app.include_router(executions.create_router(scheduler, access))
    return app


def build_api_process(settings: SchedulerSettings | None = None) -> FastAPI:
    """Wire the production API process: catalog, repository, providers, scheduler and API.

    The repository is initialized and the scheduler runs only inside the app
    lifespan.

    Args:
        settings: Process settings; None reads them from the environment.

    Returns:
        The FastAPI application served by Uvicorn.

    Raises:
        CatalogError: If the model catalog file cannot be read or is invalid.
        UnknownBuilderError: If `BUILDER_IDS` names a builder that is not registered.
        ValueError: If the provider context key is not a valid base64 key.
    """
    config = settings or SchedulerSettings()
    providers = PROVIDERS
    catalog = ModelCatalog.from_file(config.model_catalog_path)
    builder_ids = (
        frozenset(BUILDERS) if config.builder_ids is None else config.builder_ids
    )
    for builder_id in builder_ids:
        get_builder(builder_id)
    cipher = ProtectedContextCipher.from_base64(config.provider_context_key)
    repository = PostgresExecutionRepository(
        config.database_url,
        cipher=cipher,
        pool_size=config.database_pool_size,
        per_model_limit=config.per_model_limit,
        processing_policy=RetryPolicy(
            max_attempts=config.processing_attempts,
            initial_backoff=config.initial_backoff,
            maximum_backoff=config.maximum_backoff,
        ),
        finish_policy=RetryPolicy(
            max_attempts=config.provider_finish_attempts,
            initial_backoff=config.initial_backoff,
            maximum_backoff=config.maximum_backoff,
        ),
    )
    contributions = [provider.api(config) for provider in providers]
    client = httpx.AsyncClient()
    dispatch = HTTPBuilderDispatch(client, origin=config.ray_serve_origin)
    scheduler = ExecutionScheduler(
        repository=repository,
        dispatch=dispatch,
        providers={c.adapter.name: c.adapter for c in contributions},
        resolve_model=model_resolver(catalog),
        default_timeout=config.default_deadline,
        maximum_timeout=config.maximum_deadline,
        lease_duration=config.lease_duration,
        poll_interval=config.poll_interval,
        max_concurrent=config.max_concurrent_executions,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            await repository.initialize()
            async with scheduler.running():
                yield
        finally:
            await client.aclose()
            await repository.aclose()

    return create_app(
        scheduler,
        catalog=catalog,
        builder_runner=SynchronousBuilderService(catalog, dispatch),
        lifespan=lifespan,
        provider_routes=[c.routes for c in contributions if c.routes is not None],
        deployment_health=DeploymentHealth(
            client,
            origin=config.ray_serve_origin,
            routes=[f"/builders/{builder_id}" for builder_id in builder_ids]
            + [f"/{model.id}" for model in catalog],
        ),
    )
