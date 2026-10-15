"""Wire the facade process: catalog, scheduler, providers, dispatch and API."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import httpx

from model_service.builders.application import validate_model
from model_service.builders.client import HTTPBuilderDispatch
from model_service.builders.registry import BUILDERS, get_builder
from model_service.facade.api import create_app as create_api
from model_service.facade.catalog import ModelCatalog
from model_service.facade.health import DeploymentHealth
from model_service.facade.inference import HttpInferenceProxy, ModelInference
from model_service.facade.settings import SchedulerSettings
from model_service.facade.synchronous import SynchronousBuilderService
from model_service.providers.shared_storage import SharedStorageAccessAdapter
from model_service.providers.standalone import StandaloneProviderAdapter
from model_service.scheduler.domain import RetryPolicy
from model_service.scheduler.scheduler import ExecutionScheduler
from model_service.scheduler.store.postgres import PostgresExecutionRepository
from model_service.scheduler.store.protected_context import ProtectedContextCipher


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from fastapi import FastAPI

    from model_service.builders.contracts import ModelBinding
    from model_service.scheduler.contracts import ModelResolver
    from model_service.scheduler.providers import ProviderAdapter


def model_resolver(catalog: ModelCatalog) -> ModelResolver:
    def resolve(builder_id: str, model_id: str) -> ModelBinding:
        spec = get_builder(builder_id)
        model = catalog.get(model_id)
        validate_model(spec, model)
        return model

    return resolve


def create_app(settings: SchedulerSettings | None = None) -> FastAPI:
    config = settings or SchedulerSettings.from_environment()
    catalog = ModelCatalog.from_file(config.model_catalog_path)
    builder_ids = config.builder_ids or frozenset(BUILDERS)
    for builder_id in builder_ids:
        get_builder(builder_id)
    cipher = ProtectedContextCipher.from_base64(config.provider_context_key)
    repository = PostgresExecutionRepository(
        config.database_url,
        cipher=cipher,
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
    providers: dict[str, ProviderAdapter] = {
        "standalone": StandaloneProviderAdapter(
            SharedStorageAccessAdapter(config.shared_storage_roots)
        )
    }
    if config.empaia_gateway_factory is not None:
        from model_service.providers.empaia.gateway import load_gateway_factory
        from model_service.providers.empaia.provider import EmpaiaProviderAdapter

        providers["empaia"] = EmpaiaProviderAdapter(
            load_gateway_factory(config.empaia_gateway_factory),
            allowed_api_urls=config.empaia_api_urls,
            workspace_root=config.empaia_workspace_root,
        )
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
    scheduler = ExecutionScheduler(
        repository=repository,
        dispatch=dispatch,
        providers=providers,
        resolve_model=model_resolver(catalog),
        default_timeout=config.default_deadline,
        maximum_timeout=config.maximum_deadline,
        lease_duration=config.lease_duration,
        poll_interval=config.poll_interval,
        max_concurrent=config.max_concurrent_executions,
        inference_target=ModelInference(
            catalog,
            inference_proxy,
            max_result_bytes=config.inference_max_result_bytes,
        ),
        max_concurrent_inference=config.max_concurrent_inference,
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

    return create_api(
        scheduler,
        catalog=catalog,
        builder_runner=SynchronousBuilderService(catalog, dispatch),
        inference_proxy=inference_proxy,
        inference_max_request_bytes=config.inference_max_request_bytes,
        lifespan=lifespan,
        deployment_health=DeploymentHealth(
            client,
            origin=config.ray_serve_origin,
            routes=[f"/builders/{builder_id}" for builder_id in builder_ids]
            + [f"/{model.id}" for model in catalog],
        ),
    )
