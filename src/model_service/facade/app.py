"""Wire the facade process: catalog, scheduler, providers, dispatch and API."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import httpx

from model_service.builders.client import HTTPBuilderDispatch
from model_service.builders.registry import BUILDERS, get_builder
from model_service.facade.api import create_app as create_api
from model_service.facade.capacity import ServeCapacity
from model_service.facade.catalog import ModelCatalog
from model_service.facade.health import DeploymentHealth
from model_service.facade.inference import HttpInferenceProxy, ModelInference
from model_service.facade.settings import SchedulerSettings
from model_service.facade.synchronous import SynchronousBuilderService
from model_service.providers.installed import PROVIDERS
from model_service.scheduler.domain import RetryPolicy
from model_service.scheduler.scheduler import ExecutionScheduler
from model_service.scheduler.store.payloads import PayloadStore
from model_service.scheduler.store.postgres import PostgresExecutionRepository
from model_service.scheduler.store.protected_context import ProtectedContextCipher


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi import FastAPI

    from model_service.builders.contracts import ModelBinding
    from model_service.scheduler.contracts import ModelResolver


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


def create_app(settings: SchedulerSettings | None = None) -> FastAPI:
    """Wire the production facade: catalog, repository, providers, scheduler and API.

    The repository is initialized, provider background tasks start and the
    scheduler runs only inside the app lifespan.

    Args:
        settings: Process settings; None reads them from the environment.

    Returns:
        The FastAPI application served by Uvicorn.

    Raises:
        CatalogError: If the model catalog file cannot be read or is invalid.
        UnknownBuilderError: If `BUILDER_IDS` names a builder that is not registered.
        ValueError: If the provider context key is not a valid base64 key.
    """
    config = settings or SchedulerSettings.from_environment()
    providers = PROVIDERS
    catalog = ModelCatalog.from_file(
        config.model_catalog_path,
        [p.catalog_block for p in providers if p.catalog_block is not None],
    )
    builder_ids = config.builder_ids or frozenset(BUILDERS)
    for builder_id in builder_ids:
        get_builder(builder_id)
    cipher = ProtectedContextCipher.from_base64(config.provider_context_key)
    repository = PostgresExecutionRepository(
        config.database_url,
        cipher=cipher,
        payloads=PayloadStore(config.inference_storage_path),
        pool_size=config.database_pool_size,
        per_model_limit=config.per_model_limit,
        per_model_inference_limit=config.per_model_inference_limit,
        inference_retention=config.inference_retention,
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
    contributions = [provider.facade(catalog, config) for provider in providers]
    client = httpx.AsyncClient()
    dispatch = HTTPBuilderDispatch(client, origin=config.ray_serve_origin)
    inference_proxy = HttpInferenceProxy(
        client,
        origin=config.inference_serve_origin or config.ray_serve_origin,
        model_origins={
            model: str(origin)
            for model, origin in config.inference_model_origins.items()
        },
        timeout=config.inference_timeout.total_seconds(),
    )
    inference_target = ModelInference(
        catalog, inference_proxy, max_result_bytes=config.inference_max_result_bytes
    )
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
        inference_target=inference_target,
        max_concurrent_inference=config.max_concurrent_inference,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        background: list[asyncio.Task[None]] = []
        try:
            await repository.initialize()
            background = [
                asyncio.create_task(c.background())
                for c in contributions
                if c.background is not None
            ]
            async with scheduler.running():
                yield
        finally:
            for task in background:
                task.cancel()
            await asyncio.gather(*background, return_exceptions=True)
            await client.aclose()
            await repository.aclose()

    return create_api(
        scheduler,
        catalog=catalog,
        builder_runner=SynchronousBuilderService(catalog, dispatch),
        inference_target=inference_target,
        inference_max_request_bytes=config.inference_max_request_bytes,
        lifespan=lifespan,
        provider_routes=[c.routes for c in contributions if c.routes is not None],
        capacity=(
            ServeCapacity(
                client,
                origin=config.ray_dashboard_origin,
                grace_seconds=config.capacity_grace.total_seconds(),
            )
            if config.ray_dashboard_origin is not None
            else None
        ),
        deployment_health=DeploymentHealth(
            client,
            origin=config.ray_serve_origin,
            routes=[f"/builders/{builder_id}" for builder_id in builder_ids]
            + [f"/{model.id}" for model in catalog],
        ),
    )
