"""Wire production services for the standalone scheduler HTTP process.

PostgreSQL stores durable executions; a shared HTTP client reaches Ray Serve. Standalone
is always registered and EMPAIA is optional. App lifespan initializes storage and owns
worker/client shutdown.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import httpx
from fastapi import FastAPI

from execution_scheduler.domain import RetryPolicy
from execution_scheduler.persistence.postgres import PostgresExecutionRepository
from execution_scheduler.persistence.protected_context import ProtectedContextCipher
from execution_scheduler.scheduler import ExecutionScheduler
from integrations.empaia.gateway import load_gateway_factory
from integrations.empaia.provider import EmpaiaProviderAdapter
from integrations.standalone import StandaloneProviderAdapter
from model_service.api import create_execution_app
from model_service.health import WorkflowDeploymentHealth
from model_service.inference import HttpInferenceProxy
from model_service.settings import SchedulerSettings
from model_service.synchronous import SynchronousWorkflowService
from workflow_execution.catalog import WorkflowCatalog
from workflow_execution.dispatch import HTTPWorkflowDispatch


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from execution_scheduler.providers import ProviderAdapter


def create_app(settings: SchedulerSettings | None = None) -> FastAPI:
    """Construct production adapters and HTTP services from supplied settings or the environment. Connections and worker tasks begin in the app lifespan.

    Args:
        settings: Explicit runtime settings; production defaults come from environment
            variables when omitted.

    Returns:
        FastAPI: Application with registered routes and the supplied resource lifespan.

    Raises:
        ValueError: Settings, encryption key, retry bounds, or catalog defaults are
            invalid.
        ValidationError: Workflow definitions or configured inference origins fail their
            typed schemas.
        ImportError: The configured EMPAIA factory module cannot be imported.
        TypeError: The configured EMPAIA factory attribute is absent or not callable.
    """
    config = settings or SchedulerSettings.from_environment()
    catalog = WorkflowCatalog.default(config.workflow_ids)
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
    client = httpx.AsyncClient()
    dispatch = HTTPWorkflowDispatch(client, origin=config.ray_serve_origin)
    scheduler = ExecutionScheduler(
        repository=repository,
        dispatch=dispatch,
        providers=providers,
        known_workflows=catalog.identifiers,
        default_timeout=config.default_deadline,
        maximum_timeout=config.maximum_deadline,
        lease_duration=config.lease_duration,
        poll_interval=config.poll_interval,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        """Initialize the execution schema, run the scheduler during application lifetime, and close the shared HTTP client even if startup fails.

        Args:
            _app: Unused protocol argument; retained to match the interface.

        Yields:
            None: Control while application or scheduler resources remain active.

        Raises:
            ExecutionRepositoryUnavailableError: Schema initialization cannot access
                durable storage.
            ExceptionGroup: Application or worker lifetime fails inside the scheduler
                task group.
            asyncio.CancelledError: Application lifetime is cancelled; workers are
                joined and the shared HTTP client is closed.
        """
        try:
            await repository.initialize()
            async with scheduler.running():
                yield
        finally:
            await client.aclose()

    return create_execution_app(
        scheduler,
        workflow_runner=SynchronousWorkflowService(catalog, dispatch),
        inference_proxy=HttpInferenceProxy(
            client,
            origin=config.inference_serve_origin or config.ray_serve_origin,
            model_origins={
                model: str(origin)
                for model, origin in config.inference_model_origins.items()
            },
            timeout=config.inference_timeout.total_seconds(),
        ),
        lifespan=lifespan,
        deployment_health=WorkflowDeploymentHealth(
            client,
            origin=config.ray_serve_origin,
            workflow_ids=catalog.identifiers,
        ),
    )
