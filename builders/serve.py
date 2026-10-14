"""Ray Serve ingress hosting one builder, selected by `builder_id` in user_config.

Every builder application uses this module; only its user_config differs.
"""

from __future__ import annotations

import importlib
import os
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from ray import serve

from builders.application import BuilderApplication
from builders.contracts import BuilderResult
from builders.http import ErrorCode, ErrorResponse, RunRequest, error, run_http
from builders.registry import get_builder
from builders.settings import DeploymentSettings, WorkerSettings
from integrations.shared_storage import SharedStorageAccessAdapter


if TYPE_CHECKING:
    from builders.application import AccessAdapter


fastapi = FastAPI()


@fastapi.exception_handler(RequestValidationError)
async def invalid_request(
    _request: Request, _error: RequestValidationError
) -> JSONResponse:
    # Do not echo the request: it may contain provider credentials.
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={
            "detail": {
                "code": ErrorCode.INVALID_INVOCATION,
                "message": "builder invocation is invalid",
                "retryable": False,
            }
        },
    )


def build_application(config: DeploymentSettings) -> BuilderApplication:
    spec = get_builder(config.builder_id)
    module_name, _, class_name = spec.implementation.partition(":")
    implementation = getattr(importlib.import_module(module_name), class_name)
    settings = WorkerSettings.model_validate(config.model_dump(exclude={"builder_id"}))
    adapters: dict[str, AccessAdapter] = {
        "shared_storage": SharedStorageAccessAdapter(settings.shared_storage_roots)
    }
    gateway_factory = os.environ.get("EMPAIA_GATEWAY_FACTORY")
    if gateway_factory:
        from integrations.empaia.gateway import load_gateway_factory
        from integrations.empaia.input_access import EmpaiaInputAccessAdapter

        adapters["empaia"] = EmpaiaInputAccessAdapter(
            load_gateway_factory(gateway_factory),
            Path(
                os.environ.get("EMPAIA_WORKSPACE_ROOT", "/mnt/model-service-executions")
            ),
        )
    return BuilderApplication(spec, implementation(settings), adapters)


@serve.deployment(num_replicas="auto")
@serve.ingress(fastapi)
class BuilderDeployment:
    def __init__(self) -> None:
        self._application: BuilderApplication | None = None

    def reconfigure(self, values: dict[str, object]) -> None:
        self._application = build_application(DeploymentSettings.model_validate(values))

    @fastapi.post(
        "/",
        response_model=BuilderResult,
        responses={code: {"model": ErrorResponse} for code in (422, 500, 503)},
    )
    async def run(self, body: RunRequest) -> BuilderResult:
        if self._application is None:
            raise error(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                ErrorCode.UNAVAILABLE,
                "builder is not configured yet",
                retryable=True,
            )
        return await run_http(self._application, body)


app = BuilderDeployment.bind()  # type: ignore[attr-defined]
