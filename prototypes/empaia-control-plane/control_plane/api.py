from __future__ import annotations

import asyncio
import secrets
import sqlite3
from contextlib import asynccontextmanager
from datetime import datetime  # noqa: TC003 - Pydantic resolves this at runtime.
from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, StringConstraints

from control_plane.app_service import AppServiceClient, FileTokenProvider
from control_plane.domain import (
    AdmissionRequest,
    Execution,
    ExecutionNotFoundError,
    ExecutionState,
    FailureCategory,
    SubmissionConflictError,
    TimeoutKind,
)
from control_plane.reconciler import ExecutionReconciler
from control_plane.registry import AppRegistry
from control_plane.resolver import EmpaiaExecutionResolver
from control_plane.service import (
    ExecutionService,
    InstallationAliasError,
    UnknownAppMappingError,
)
from control_plane.storage import StorageLocatorResolver
from control_plane.store import SQLiteExecutionStore


if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from control_plane.settings import Settings


Identifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=255,
        pattern=r"^[^\s\x00-\x1f\x7f]+$",
    ),
]
IdempotencyKey = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=16,
        max_length=255,
        pattern=r"^[^\s\x00-\x1f\x7f]+$",
    ),
]
_JES_BEARER = HTTPBearer(auto_error=False, scheme_name="JESBearer")


class SubmissionBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    empaia_job_id: Identifier
    registered_app_id: Identifier
    registered_app_version: Identifier
    installation_alias: Identifier
    idempotency_key: IdempotencyKey


class SubmissionResponse(BaseModel):
    execution_id: str
    state: ExecutionState
    status_url: str


class SafeReason(BaseModel):
    code: str
    message: str


class ExecutionStatusResponse(BaseModel):
    execution_id: str
    empaia_job_id: str
    state: ExecutionState
    cancellation_requested: bool
    created_at: datetime
    updated_at: datetime
    ready_at: datetime | None
    first_claimed_at: datetime | None
    failure_category: FailureCategory | None
    timeout_kind: TimeoutKind | None
    reason: SafeReason | None


def create_app(settings: Settings) -> FastAPI:
    _validate_distinct_credentials(settings)
    store = SQLiteExecutionStore(settings.database_path)
    service = ExecutionService(
        store=store,
        registry=AppRegistry(settings.app_mappings),
        installation_alias=settings.installation_alias,
    )
    reconciler = ExecutionReconciler(
        store=store,
        max_initial_capacity_wait_seconds=(settings.max_initial_capacity_wait_seconds),
        worker_stale_after_seconds=settings.worker_stale_after_seconds,
        retry_backoff_seconds=settings.retry_backoff_seconds,
    )
    resolver = _build_resolver(settings, store)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        stop = asyncio.Event()
        tasks = [
            asyncio.create_task(
                reconciler.run(stop, settings.reconcile_interval_seconds),
                name="execution-reconciler",
            )
        ]
        tasks.append(
            asyncio.create_task(
                resolver.run(stop, settings.reconcile_interval_seconds),
                name="execution-resolver",
            )
        )
        try:
            yield
        finally:
            stop.set()
            await asyncio.gather(*tasks)

    app = FastAPI(
        title="RationAI model-service EMPAIA control plane",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.execution_store = store

    def authenticate(
        credentials: Annotated[
            HTTPAuthorizationCredentials | None, Depends(_JES_BEARER)
        ],
    ) -> None:
        authenticated = (
            credentials is not None
            and credentials.scheme.lower() == "bearer"
            and secrets.compare_digest(
                credentials.credentials,
                _read_jes_token(settings),
            )
        )
        if not authenticated:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid service credential",
                headers={"WWW-Authenticate": "Bearer"},
            )

    authenticated = [Depends(authenticate)]

    @app.exception_handler(sqlite3.Error)
    async def database_error_handler(
        _request: Request, _error: sqlite3.Error
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "durable execution store is unavailable"},
            headers={"Retry-After": "1"},
        )

    @app.get("/healthz", include_in_schema=False)
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    def readiness() -> dict[str, str]:
        store.check_ready()
        return {"status": "ready"}

    @app.post(
        "/v1/empaia/executions",
        response_model=SubmissionResponse,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=authenticated,
    )
    def submit(body: SubmissionBody, response: Response) -> SubmissionResponse:
        request = AdmissionRequest(
            installation_alias=body.installation_alias,
            empaia_job_id=body.empaia_job_id,
            registered_app_id=body.registered_app_id,
            registered_app_version=body.registered_app_version,
            idempotency_key=body.idempotency_key,
        )
        try:
            result = service.admit(request)
        except SubmissionConflictError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except (InstallationAliasError, UnknownAppMappingError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        response.headers["Location"] = _status_url(result.execution.id)
        return SubmissionResponse(
            execution_id=result.execution.id,
            state=result.execution.state,
            status_url=_status_url(result.execution.id),
        )

    @app.get(
        "/v1/empaia/executions/{execution_id}",
        response_model=ExecutionStatusResponse,
        dependencies=authenticated,
    )
    def get_status(execution_id: str) -> ExecutionStatusResponse:
        try:
            execution = service.get(execution_id)
            return _status_response(execution)
        except ExecutionNotFoundError as error:
            raise HTTPException(
                status_code=404, detail="execution not found"
            ) from error

    @app.post(
        "/v1/empaia/executions/{execution_id}/cancellation",
        response_model=ExecutionStatusResponse,
        status_code=status.HTTP_202_ACCEPTED,
        dependencies=authenticated,
    )
    def cancel(execution_id: str) -> ExecutionStatusResponse:
        try:
            execution = service.cancel(execution_id)
            return _status_response(execution)
        except ExecutionNotFoundError as error:
            raise HTTPException(
                status_code=404, detail="execution not found"
            ) from error

    return app


def _status_url(execution_id: str) -> str:
    return f"/v1/empaia/executions/{execution_id}"


def _status_response(execution: Execution) -> ExecutionStatusResponse:
    reason = None
    if execution.safe_reason_code and execution.safe_reason_message:
        reason = SafeReason(
            code=execution.safe_reason_code,
            message=execution.safe_reason_message,
        )
    return ExecutionStatusResponse(
        execution_id=execution.id,
        empaia_job_id=execution.empaia_job_id,
        state=execution.state,
        cancellation_requested=execution.cancellation_requested,
        created_at=execution.created_at,
        updated_at=execution.updated_at,
        ready_at=execution.ready_at,
        first_claimed_at=execution.first_claimed_at,
        failure_category=execution.failure_category,
        timeout_kind=execution.timeout_kind,
        reason=reason,
    )


def _build_resolver(
    settings: Settings, store: SQLiteExecutionStore
) -> EmpaiaExecutionResolver:
    if not settings.storage_mappings:
        raise ValueError(
            "at least one storage mapping is required while admission is enabled"
        )
    if any(not mapping.allowed_wsi_plugin_ids for mapping in settings.app_mappings):
        raise ValueError("every enabled app must allow a supported WSI plugin")
    if not settings.allow_broad_app_service_credential:
        raise ValueError(
            "the current POC resolver requires explicit acceptance of its broad "
            "AppService credential boundary"
        )
    return EmpaiaExecutionResolver(
        store=store,
        app_service=AppServiceClient(
            settings.app_service_url,
            FileTokenProvider(
                settings.app_service_token_file,
                forbidden_paths=(settings.jes_bearer_token_file,),
            ),
            allow_insecure_http=settings.allow_insecure_app_service_http,
        ),
        storage=StorageLocatorResolver(settings.storage_mappings),
        retry_seconds=settings.resolution_retry_seconds,
    )


def _read_api_token(path: Path) -> str:
    try:
        return _read_token(path)
    except RuntimeError as error:
        raise HTTPException(
            status_code=503, detail="service credential is temporarily unavailable"
        ) from error


def _read_jes_token(settings: Settings) -> str:
    jes_token = _read_api_token(settings.jes_bearer_token_file)
    app_service_token = _read_api_token(settings.app_service_token_file)
    if secrets.compare_digest(jes_token, app_service_token):
        raise HTTPException(
            status_code=503, detail="service credential isolation is invalid"
        )
    return jes_token


def _validate_distinct_credentials(settings: Settings) -> None:
    jes_token = _read_token(settings.jes_bearer_token_file)
    app_service_token = _read_token(settings.app_service_token_file)
    if secrets.compare_digest(jes_token, app_service_token):
        raise ValueError("JES and AppService credentials must be distinct")


def _read_token(path: Path) -> str:
    try:
        token = path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise RuntimeError("credential file is unavailable") from error
    if not token:
        raise RuntimeError("credential file is empty")
    return token
