"""Build the API process.

create_app assembles the FastAPI app from its routes; build_api_process wires
the catalog, scheduler, providers and dispatch into it.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import httpx
from asgi_correlation_id import CorrelationIdMiddleware
from fastapi import FastAPI, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from model_service.api.catalog_mount import (
    CatalogState,
    CatalogStatus,
    MountedCatalog,
)
from model_service.api.health import (
    DeploymentHealth,
    DeploymentState,
    DeploymentStatus,
)
from model_service.api.http.auth import Access
from model_service.api.http.errors import ApiErrorCode, http_error
from model_service.api.routes.v1 import analyses, executions, models
from model_service.api.scheduler.dispatch import HTTPAnalysisDispatch
from model_service.api.scheduler.execution_scheduler import ExecutionScheduler
from model_service.api.scheduler.internal_state import RetryPolicy
from model_service.api.scheduler.public import SchedulingUnavailableError
from model_service.api.scheduler.settings import SchedulerSettings
from model_service.api.scheduler.store.postgres.repository import (
    PostgresExecutionRepository,
)
from model_service.api.scheduler.store.protected_context import ProtectedContextCipher
from model_service.api.scheduler.targets import CapacityKind
from model_service.api.synchronous_runs import SynchronousAnalysisService
from model_service.catalog.analyses import FOLDER as ANALYSES
from model_service.catalog.analyses import resolve_run
from model_service.catalog.served_models import FOLDER as SERVED_MODELS
from model_service.providers.registry import PROVIDERS
from model_service.providers.standalone.provider import STANDALONE_PROVIDER


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Sequence

    from starlette.types import Lifespan

    from model_service.api.scheduler.protocols import Scheduler
    from model_service.api.scheduler.public import AnalysisResolver
    from model_service.api.scheduler.targets import CapacityKey, CapacityLimits
    from model_service.catalog import Catalog
    from model_service.catalog.analyses import ResolvedAnalysis
    from model_service.contracts.json_values import JsonObject
    from model_service.providers.provider import RouteFactory


def analysis_resolver(catalog: Callable[[], Catalog]) -> AnalysisResolver:
    """Make the scheduler's resolver of a deployed analysis with a run's settings.

    Args:
        catalog: Gives the catalog in use; it may reload between admissions.

    Returns:
        A function giving the resolved analysis, raising `UnknownAnalysisError`
        or `InvalidInvocation`.
    """

    def resolve(analysis_id: str, settings: JsonObject | None) -> ResolvedAnalysis:
        """Resolve an analysis of the current catalog.

        Args:
            analysis_id: The analysis.
            settings: Per-run settings; None keeps the file's.

        Returns:
            The analysis and its document.
        """
        return resolve_run(catalog(), analysis_id, settings)

    return resolve


def capacity_limits(catalog: Callable[[], Catalog], default: int) -> CapacityLimits:
    """Make the scheduler's slots per capacity key.

    A served model's `serving.max_concurrent_runs` sets its slots; a model
    without it, one no longer deployed and a model-free analysis get the
    default.

    Args:
        catalog: Gives the catalog in use; it may reload between claims.
        default: Slots where the catalog sets none (`PER_MODEL_LIMIT`).

    Returns:
        The limits.
    """

    def limit(key: CapacityKey) -> int:
        """Slots of one capacity key in the current catalog.

        Args:
            key: A served model or a model-free analysis.

        Returns:
            How many attempts may hold a slot at once.
        """
        if key.kind is CapacityKind.MODEL:
            entry = catalog().entries(SERVED_MODELS).get(key.id)
            if entry is not None and entry.value.serving.max_concurrent_runs:
                return int(entry.value.serving.max_concurrent_runs)
        return default

    return limit


def create_app(
    scheduler: Scheduler,
    *,
    catalog: Callable[[], Catalog],
    analysis_runner: SynchronousAnalysisService | None = None,
    lifespan: Lifespan[FastAPI] | None = None,
    deployment_health: DeploymentHealth | None = None,
    catalog_status: Callable[[], CatalogStatus] | None = None,
    access: Access | None = None,
    provider_routes: Sequence[RouteFactory] = (),
) -> FastAPI:
    """Build the FastAPI app with health checks, public routes and provider routes.

    Args:
        scheduler: Admits, reads and cancels durable executions.
        catalog: Gives the catalog in use: served models under `/v1/models`,
            analyses under `/v1/analyses`.
        analysis_runner: Runs analyses synchronously; None makes
            `/v1/analyses/{id}/run` answer 503.
        lifespan: Startup and shutdown context for the app.
        deployment_health: Checks Ray Serve routes; None reports `not_configured`.
        catalog_status: Whether the mounted catalog is the one in use; None
            for a catalog that never reloads, always `loaded`.
        access: Guards the public API; the default lets everyone in. Providers
            guard their own routes.
        provider_routes: Router factories contributed by providers.

    Returns:
        The assembled application.
    """
    access = access or Access()
    app = FastAPI(title="model-service", lifespan=lifespan)
    # Reads `X-Request-ID` or makes one, returns it, and names it in log lines.
    # Any caller's ID is kept: it only labels the request.
    app.add_middleware(CorrelationIdMiddleware, validator=None)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(
        _request: Request, error: RequestValidationError
    ) -> JSONResponse:
        """Answer a request that fails validation without echoing its body.

        Args:
            _request: The rejected request.
            error: The validation error.

        Returns:
            A 422 response with the `invalid_request` error envelope.
        """
        # Do not echo the body: it may contain provider credentials.
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={
                "detail": {
                    "code": ApiErrorCode.INVALID_REQUEST,
                    "message": _invalid_body_message(error),
                    "retryable": False,
                }
            },
        )

    @app.get("/healthz")
    async def health() -> JSONResponse:
        """Report liveness: unhealthy once the scheduler's worker has stopped.

        Returns:
            200 while the worker runs, 503 after it stopped.
        """
        # Liveness: an API process whose worker stopped would hold jobs forever.
        if not scheduler.alive:
            return JSONResponse({"status": "worker stopped"}, status_code=503)
        return JSONResponse({"status": "ok"})

    @app.get("/readyz")
    async def readiness() -> dict[str, str]:
        """Report readiness: whether the execution repository answers.

        Returns:
            `{"status": "ok"}` when ready.

        Raises:
            HTTPException: 503 `scheduling_unavailable` when the repository does
                not answer.
        """
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
        """Report whether Ray Serve exposes every served-model and runner route.

        Args:
            response: The response, set to 503 when routes are missing or Ray
                Serve is unreachable.

        Returns:
            The deployment state and any missing routes.
        """
        if deployment_health is None:
            return DeploymentStatus(DeploymentState.NOT_CONFIGURED)
        result = await deployment_health.check()
        if result.status is not DeploymentState.READY:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return result

    @app.get("/healthz/catalog")
    async def catalog_health(response: Response) -> CatalogStatus:
        """Report whether the mounted catalog loaded, or why the API kept an older one.

        Args:
            response: The response, set to 503 after a failed reload.

        Returns:
            The catalog's state and the reload error, if any.
        """
        result = (
            catalog_status() if catalog_status else CatalogStatus(CatalogState.LOADED)
        )
        if result.status is not CatalogState.LOADED:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return result

    app.include_router(models.create_router(catalog, access))
    app.include_router(
        analyses.create_router(
            catalog, analysis_runner, scheduler, access, STANDALONE_PROVIDER
        )
    )
    for routes in provider_routes:
        app.include_router(routes(scheduler))
    app.include_router(executions.create_router(scheduler, access))
    return app


def build_api_process(settings: SchedulerSettings | None = None) -> FastAPI:
    """Wire the production API process: catalog, repository, providers, scheduler and API.

    The repository is initialized, provider background tasks start and the
    scheduler runs only inside the app lifespan.

    Args:
        settings: Process settings; None reads them from the environment.

    Returns:
        The FastAPI application served by Uvicorn.

    Raises:
        CatalogError: If the mounted catalog does not load.
        ValueError: If the provider context key is not a valid base64 key.
    """
    config = settings or SchedulerSettings()
    providers = PROVIDERS
    mounted = MountedCatalog(config.model_catalog_dir)

    def catalog() -> Catalog:
        """The catalog in use; a reload swaps it."""
        return mounted.resolved

    cipher = ProtectedContextCipher.from_base64(config.provider_context_key)
    repository = PostgresExecutionRepository(
        config.database_url,
        cipher=cipher,
        pool_size=config.database_pool_size,
        capacity_limits=capacity_limits(catalog, config.per_model_limit),
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
    contributions = [provider.api(catalog, config) for provider in providers]
    client = httpx.AsyncClient()
    dispatch = HTTPAnalysisDispatch(client, origin=config.ray_serve_origin)
    scheduler = ExecutionScheduler(
        repository=repository,
        dispatch=dispatch,
        providers={c.adapter.name: c.adapter for c in contributions},
        resolve_analysis=analysis_resolver(catalog),
        default_timeout=config.default_deadline,
        maximum_timeout=config.maximum_deadline,
        lease_duration=config.lease_duration,
        poll_interval=config.poll_interval,
        max_concurrent=config.max_concurrent_executions,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        """Run the scheduler and provider background work for the app's lifetime.

        Initializes the repository first, and on shutdown cancels background
        work and closes the HTTP client and the repository.

        Args:
            _app: The FastAPI app.

        Yields:
            Control while the app serves requests.
        """
        background: list[asyncio.Task[None]] = []
        try:
            await repository.initialize()
            background = [
                asyncio.create_task(c.background())
                for c in contributions
                if c.background is not None
            ]
            background.append(
                asyncio.create_task(mounted.watch(config.catalog_reload_seconds))
            )
            async with scheduler.running():
                yield
        finally:
            for task in background:
                task.cancel()
            await asyncio.gather(*background, return_exceptions=True)
            await client.aclose()
            await repository.aclose()

    return create_app(
        scheduler,
        catalog=catalog,
        analysis_runner=SynchronousAnalysisService(catalog, dispatch),
        lifespan=lifespan,
        provider_routes=[c.routes for c in contributions if c.routes is not None],
        deployment_health=DeploymentHealth(
            client,
            origin=config.ray_serve_origin,
            routes=lambda: (
                [f"/{model_id}" for model_id in catalog().entries(SERVED_MODELS)]
                + (["/analyses"] if catalog().entries(ANALYSES) else [])
            ),
        ),
        catalog_status=mounted.status,
    )


def _invalid_body_message(error: RequestValidationError) -> str:
    """Name the failing fields and why, without the values sent.

    Args:
        error: The validation error.

    Returns:
        The message, e.g. `request body is invalid: deadline_at: ...`.
    """
    # Pydantic's `msg` describes the rule; the rejected value is only in `input`.
    problems = [
        f"{'.'.join(str(part) for part in item['loc'][1:]) or 'body'}: {item['msg']}"
        for item in error.errors()[:5]
    ]
    if not problems:
        return "request body is invalid"
    return f"request body is invalid: {'; '.join(problems)}"
