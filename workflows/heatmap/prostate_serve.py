"""Expose the prostate workflow as a Ray Serve HTTP application.

Serve reconfiguration wires slide processing to the classifier's Ray handle and optional
EMPAIA input access. This replica owns computation, not the durable scheduler or
database.
"""

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
from workflow_execution.contracts import WorkflowResult
from workflow_execution.http import (
    WorkflowErrorResponse,
    WorkflowHTTPErrorCode,
    WorkflowRunRequest,
    run_workflow_http,
)
from workflows.heatmap.processor import SlideHeatmapProcessor
from workflows.heatmap.prostate import (
    ProstateHeatmapWorkflow,
)
from workflows.heatmap.ray_predictor import RayHeatmapPredictor


class Config(TypedDict):
    """Declare thread and tile-concurrency settings supplied by Ray Serve user configuration.

    Attributes:
        num_threads: Number of synchronous slide-read threads.
        max_concurrent_tasks: Maximum in-flight tile read/prediction tasks.
    """

    num_threads: int
    max_concurrent_tasks: int


fastapi = FastAPI()


@fastapi.exception_handler(RequestValidationError)
async def invalid_request(
    _request: Request,
    _error: RequestValidationError,
) -> JSONResponse:
    """Return a safe 422 validation envelope without reflecting the internal request or protected context.

    Args:
        _request: Unused request to validate, admit, dispatch, forward, or process, as
            specified by its declared type; retained to match the interface.
        _error: Unused safe domain error or injected integration exception to classify;
            retained to match the interface.

    Returns:
        JSONResponse: Safe JSON validation or availability response.
    """
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
    """Host one configured workflow application behind Ray Serve's internal ingress."""

    def __init__(self) -> None:
        """Start unconfigured until Ray Serve supplies user configuration."""
        self._application: WorkflowApplication | None = None

    def reconfigure(self, config: Config) -> None:
        """Build the processor, classifier handle adapter, workflow wrapper, and optional provider access from configuration/environment.

        Args:
            config: Ray Serve user configuration for read-thread and tile-concurrency
                limits.

        Returns:
            None: No return value.
        """
        catalog = WorkflowCatalog.default()
        definition = catalog.get("prostate-heatmap")
        processor = SlideHeatmapProcessor(
            RayHeatmapPredictor("prostate-classifier-1"),
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
            catalog=catalog,
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
    async def run(self, body: WorkflowRunRequest) -> WorkflowResult:
        """Reject an unconfigured replica or execute the internal request through the shared safe HTTP adapter.

        Args:
            body: Internal invocation envelope including private processing attempt
                context.

        Returns:
            WorkflowResult: Named inline outputs and/or references to completed artifact
                files.

        Raises:
            HTTPException: The request fails the route's safe validation, availability,
                or execution contract.
        """
        if self._application is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": WorkflowHTTPErrorCode.UNAVAILABLE,
                    "message": "workflow application is not configured",
                    "retryable": True,
                },
            )
        return await run_workflow_http(self._application, body, body.execution_context)


app = ProstateHeatmapApplication.bind()  # type: ignore[attr-defined]
