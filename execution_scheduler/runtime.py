from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING
from uuid import uuid4

import httpx
from fastapi import FastAPI

from execution_scheduler.active_work import ActiveWorkRegistry
from execution_scheduler.api import create_execution_app
from execution_scheduler.dispatch import HTTPWorkflowDispatch
from execution_scheduler.domain import RetryPolicy
from execution_scheduler.health import WorkflowDeploymentHealth
from execution_scheduler.inference import HttpInferenceProxy
from execution_scheduler.postgres import PostgresExecutionRepository
from execution_scheduler.scheduler import ExecutionScheduler
from execution_scheduler.security import ProtectedContextCipher
from execution_scheduler.settings import SchedulerSettings
from execution_scheduler.standalone import StandaloneProviderAdapter
from execution_scheduler.synchronous import SynchronousWorkflowService
from execution_scheduler.worker import ExecutionWorker
from integrations.empaia.gateway import load_gateway_factory
from integrations.empaia.provider import EmpaiaProviderAdapter
from workflow_execution.catalog import WorkflowCatalog


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from execution_scheduler.providers import ProviderAdapter


def create_app(settings: SchedulerSettings | None = None) -> FastAPI:
    config = settings or SchedulerSettings.from_environment()
    catalog = WorkflowCatalog.default()
    cipher = ProtectedContextCipher.from_base64(config.provider_context_key)
    processing_policy = RetryPolicy(
        max_attempts=config.processing_attempts,
        initial_backoff=config.initial_backoff,
        maximum_backoff=config.maximum_backoff,
    )
    finish_policy = RetryPolicy(
        max_attempts=config.provider_finish_attempts,
        initial_backoff=config.initial_backoff,
        maximum_backoff=config.maximum_backoff,
    )
    repository = PostgresExecutionRepository(
        config.database_url,
        cipher=cipher,
        per_workflow_limit=config.per_workflow_limit,
        processing_policy=processing_policy,
        finish_policy=finish_policy,
    )
    standalone_provider = StandaloneProviderAdapter(catalog)
    providers: dict[str, ProviderAdapter] = {
        standalone_provider.name: standalone_provider
    }
    if config.empaia_gateway_factory is not None:
        gateway_factory = load_gateway_factory(config.empaia_gateway_factory)
        empaia_provider = EmpaiaProviderAdapter(catalog, gateway_factory)
        providers[empaia_provider.name] = empaia_provider
    active_work = ActiveWorkRegistry()
    client = httpx.AsyncClient()
    dispatch = HTTPWorkflowDispatch(client, origin=config.ray_serve_origin)
    scheduler = ExecutionScheduler(
        repository=repository,
        providers=providers,
        known_workflows=catalog.identifiers,
        default_timeout=config.default_deadline,
        maximum_timeout=config.maximum_deadline,
        active_work=active_work,
    )
    worker = ExecutionWorker(
        repository=repository,
        providers=providers,
        dispatch=dispatch,
        owner=f"execution-scheduler-{uuid4()}",
        lease_duration=config.lease_duration,
        active_work=active_work,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        await repository.initialize()
        stop = asyncio.Event()
        worker_task = asyncio.create_task(
            worker.run(stop, poll_interval=config.poll_interval)
        )
        try:
            yield
        finally:
            stop.set()
            worker_task.cancel()
            await asyncio.gather(worker_task, return_exceptions=True)
            await client.aclose()

    return create_execution_app(
        scheduler,
        workflow_runner=SynchronousWorkflowService(catalog, dispatch),
        inference_proxy=HttpInferenceProxy(
            client,
            origin=config.inference_serve_origin or config.ray_serve_origin,
            timeout=config.inference_timeout.total_seconds(),
        ),
        lifespan=lifespan,
        deployment_health=WorkflowDeploymentHealth(
            client,
            origin=config.ray_serve_origin,
            workflow_ids=catalog.identifiers,
        ),
    )
