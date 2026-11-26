"""The production FastAPI application Uvicorn imports.

Uvicorn configures logging before importing it, from `log-config.json` next to
this module (`--log-config`): JSON lines on stdout, each with its request id.
Executions live in PostgreSQL; pipeline runs and route health go to Ray Serve
over one HTTP client.
"""

import httpx

from model_service.control_plane.app_factory import build_control_plane
from model_service.control_plane.scheduler.settings import SchedulerSettings
from model_service.control_plane.scheduler.store.postgres.storage import (
    PostgresStorage,
)


settings = SchedulerSettings()
app = build_control_plane(
    settings,
    storage=PostgresStorage(
        settings.database_url, pool_size=settings.database_pool_size
    ),
    client=httpx.AsyncClient(),
)
