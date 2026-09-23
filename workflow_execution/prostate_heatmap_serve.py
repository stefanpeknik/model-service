from __future__ import annotations

import os
from pathlib import Path
from typing import TypedDict

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from ray import serve

from workflow_execution.application import WorkflowApplication
from workflow_execution.catalog import WorkflowCatalog
from workflow_execution.contracts import (
    WorkflowInvocation,
    WorkflowResult,
)
from workflow_execution.http import (
    WorkflowErrorResponse,
    WorkflowHTTPErrorCode,
    run_workflow_http,
)
from workflow_execution.prostate_heatmap import (
    ProstateHeatmapWorkflow,
    RayProstateHeatmapProcessor,
)


class Config(TypedDict):
    num_threads: int
    max_concurrent_tasks: int


fastapi = FastAPI()


@fastapi.exception_handler(RequestValidationError)
async def invalid_request(
    _request: Request,
    _error: RequestValidationError,
) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={
            "detail": {
                "code": WorkflowHTTPErrorCode.INVALID_INVOCATION,
                "message": "workflow invocation is invalid",
                "retryable": False,
            }
        },
    )


@serve.deployment(num_replicas="auto")
@serve.ingress(fastapi)
class ProstateHeatmapApplication:
    def __init__(self) -> None:
        self._application: WorkflowApplication | None = None

    def reconfigure(self, config: Config) -> None:
        catalog = WorkflowCatalog.default()
        definition = catalog.get("prostate-heatmap")
        processor = RayProstateHeatmapProcessor(
            num_threads=config["num_threads"],
            max_concurrent_tasks=config["max_concurrent_tasks"],
        )
        access_adapters = None
        gateway_factory_path = os.environ.get("EMPAIA_GATEWAY_FACTORY")
        if gateway_factory_path:
            from integrations.empaia.gateway import load_gateway_factory
            from integrations.empaia.input_access import EmpaiaInputAccessAdapter

            workspace_root = Path(
                os.environ.get(
                    "EMPAIA_WORKSPACE_ROOT",
                    "/mnt/model-service-executions",
                )
            )
            access_adapters = {
                "empaia": EmpaiaInputAccessAdapter(
                    load_gateway_factory(gateway_factory_path),
                    workspace_root,
                )
            }
        self._application = WorkflowApplication(
            definition,
            ProstateHeatmapWorkflow(processor),
            access_adapters=access_adapters,
        )

    @fastapi.post(
        "/",
        response_model=WorkflowResult,
        responses={
            422: {"model": WorkflowErrorResponse},
            500: {"model": WorkflowErrorResponse},
            503: {"model": WorkflowErrorResponse},
        },
    )
    async def run(self, body: WorkflowInvocation) -> WorkflowResult:
        if self._application is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": WorkflowHTTPErrorCode.UNAVAILABLE,
                    "message": "workflow application is not configured",
                    "retryable": True,
                },
            )
        return await run_workflow_http(self._application, body)


app = ProstateHeatmapApplication.bind()  # type: ignore[attr-defined]
