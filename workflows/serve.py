from __future__ import annotations

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import Field
from ray import serve

from workflow_execution.application import WorkflowApplication
from workflow_execution.catalog import WorkflowCatalog
from workflow_execution.contracts import WorkflowResult
from workflow_execution.http import (
    WorkflowHTTPErrorCode,
    WorkflowRunRequest,
    run_workflow_http,
)
from workflows.registry import build_workflow
from workflows.settings import WorkflowWorkerSettings


class DeploymentSettings(WorkflowWorkerSettings):
    workflow_id: str = Field(min_length=1)


fastapi = FastAPI()


@fastapi.exception_handler(RequestValidationError)
async def invalid_request(
    _request: Request, _error: RequestValidationError
) -> JSONResponse:
    return JSONResponse(
        status_code=422,
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
class WorkflowApplicationDeployment:
    def __init__(self) -> None:
        self._application: WorkflowApplication | None = None

    def reconfigure(self, values: dict[str, object]) -> None:
        config = DeploymentSettings.model_validate(values)
        settings = WorkflowWorkerSettings.model_validate(
            config.model_dump(exclude={"workflow_id"})
        )
        self._application = WorkflowApplication(
            WorkflowCatalog.default().get(config.workflow_id),
            build_workflow(config.workflow_id, settings),
        )

    @fastapi.post("/", response_model=WorkflowResult)
    async def run(self, body: WorkflowRunRequest) -> WorkflowResult:
        if self._application is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "code": WorkflowHTTPErrorCode.UNAVAILABLE,
                    "message": "workflow application is not configured",
                    "retryable": True,
                },
            )
        return await run_workflow_http(self._application, body, body.execution_context)


app = WorkflowApplicationDeployment.bind()  # type: ignore[attr-defined]
