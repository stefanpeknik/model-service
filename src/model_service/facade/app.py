"""Wire the facade process: catalog, scheduler, providers, dispatch and API."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import httpx

from model_service.builders.application import validate_model
from model_service.builders.client import HTTPBuilderDispatch
from model_service.builders.registry import BUILDERS, get_builder
from model_service.facade.api import create_app as create_api
from model_service.facade.capacity import ServeCapacity
from model_service.facade.catalog import ModelCatalog
from model_service.facade.health import DeploymentHealth
from model_service.facade.inference import HttpInferenceProxy, ModelInference
from model_service.facade.settings import SchedulerSettings
from model_service.facade.synchronous import SynchronousBuilderService
from model_service.providers.composition import load_providers
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
    def resolve(builder_id: str, model_id: str) -> ModelBinding:
        spec = get_builder(builder_id)
        model = catalog.get(model_id)
        validate_model(spec, model)
        return model

    return resolve


def create_app(settings: SchedulerSettings | None = None) -> FastAPI:
    config = settings or SchedulerSettings.from_environment()
    providers = load_providers()
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
