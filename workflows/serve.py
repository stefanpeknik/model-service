"""Host image-local standalone workflows without durable scheduler responsibilities."""

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
    """Select the workflow hosted by one Ray ingress deployment.

    Attributes:
        workflow_id: Built-in segmentation or embedding workflow to construct.
    """

    workflow_id: str = Field(min_length=1)


fastapi = FastAPI()


@fastapi.exception_handler(RequestValidationError)
async def invalid_request(
    _request: Request, _error: RequestValidationError
) -> JSONResponse:
    """Return a safe validation error without reflecting protected request fields.

    Args:
        _request: Invalid incoming request, intentionally excluded from the response.
        _error: Validation details, intentionally excluded from the response.

    Returns:
        JSONResponse: Permanent 422 error using the internal workflow schema.
    """
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
    """Own workflow construction and its HTTP execution adapter inside a Ray replica."""

    def __init__(self) -> None:
        """Wait for Ray reconfiguration before accepting execution requests."""
        self._application: WorkflowApplication | None = None

    def reconfigure(self, values: dict[str, object]) -> None:
        """Replace computation wiring with validated deployment settings.

        Args:
            values: Workflow identifier, tile limits, and internal raw-model origin.

        Returns:
            None: The next request uses the constructed workflow application.

        Raises:
            ValidationError: Deployment settings do not match the typed schema.
            KeyError: The selected workflow has no construction registration.
        """
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
        """Execute the internal request using shared safe HTTP error translation.

        Args:
            body: Invocation and private scheduler processing-attempt context.

        Returns:
            WorkflowResult: Named completed artifacts produced by the workflow.

        Raises:
            HTTPException: The replica is unconfigured or the invocation fails.
            asyncio.CancelledError: The request's owning task cancels computation.
        """
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
