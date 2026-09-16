from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from ray import serve

from workflow_execution.application import WorkflowApplication
from workflow_execution.catalog import WorkflowCatalog
from workflow_execution.contracts import (
    InvalidWorkflowInvocation,
    WorkflowContractViolation,
    WorkflowFailure,
    WorkflowInvocation,
    WorkflowResult,
)
from workflow_execution.prostate_heatmap import (
    ProstateHeatmapWorkflow,
    RayProstateHeatmapProcessor,
)


class Config(TypedDict):
    num_threads: int
    max_concurrent_tasks: int


@dataclass(frozen=True, slots=True)
class WorkflowErrorDetail:
    code: str
    message: str
    retryable: bool


@dataclass(frozen=True, slots=True)
class WorkflowErrorResponse:
    detail: WorkflowErrorDetail


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
                "code": "invalid_workflow_invocation",
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
        self._application = WorkflowApplication(
            definition,
            ProstateHeatmapWorkflow(processor),
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
                    "code": "workflow_unavailable",
                    "message": "workflow application is not configured",
                    "retryable": True,
                },
            )
        try:
            return await self._application.run(body)
        except InvalidWorkflowInvocation as error:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail={
                    "code": "invalid_workflow_invocation",
                    "message": str(error),
                    "retryable": False,
                },
            ) from error
        except WorkflowContractViolation as error:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={
                    "code": "workflow_contract_violation",
                    "message": "deployed workflow violated its contract",
                    "retryable": False,
                },
            ) from error
        except WorkflowFailure as error:
            raise HTTPException(
                status_code=(
                    status.HTTP_503_SERVICE_UNAVAILABLE
                    if error.retryable
                    else status.HTTP_422_UNPROCESSABLE_CONTENT
                ),
                detail={
                    "code": error.code,
                    "message": error.safe_message,
                    "retryable": error.retryable,
                },
            ) from error


app = ProstateHeatmapApplication.bind()  # type: ignore[attr-defined]
