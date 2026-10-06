from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import FastAPI
from PIL import Image
from ray import serve

from workflow_execution.application import WorkflowApplication
from workflow_execution.catalog import WorkflowCatalog
from workflow_execution.contracts import WorkflowResult
from workflow_execution.http import (
    WorkflowErrorResponse,
    WorkflowRunRequest,
    run_workflow_http,
)
from workflows.heatmap.prostate import ProstateHeatmapWorkflow


if TYPE_CHECKING:
    from workflows.heatmap.processor import HeatmapBuildRequest


class ValidationHeatmapProcessor:
    async def build(self, request: HeatmapBuildRequest) -> None:
        request.slide_path.read_bytes()
        request.tissue_mask_path.read_bytes()
        if "cancellation" in request.output_path.parts:
            await asyncio.sleep(300)
        Image.new("L", (8, 8), color=128).save(request.output_path, format="TIFF")


fastapi = FastAPI()


@serve.deployment(num_replicas="auto")
@serve.ingress(fastapi)
class ValidationProstateHeatmapApplication:
    def __init__(self) -> None:
        definition = WorkflowCatalog.default().get("prostate-heatmap")
        access_adapters = None
        gateway_factory_path = os.environ.get("EMPAIA_GATEWAY_FACTORY")
        if gateway_factory_path:
            from integrations.empaia.gateway import load_gateway_factory
            from integrations.empaia.input_access import EmpaiaInputAccessAdapter

            access_adapters = {
                "empaia": EmpaiaInputAccessAdapter(
                    load_gateway_factory(gateway_factory_path),
                    Path(os.environ["EMPAIA_WORKSPACE_ROOT"]),
                )
            }
        self._application = WorkflowApplication(
            definition,
            ProstateHeatmapWorkflow(ValidationHeatmapProcessor()),
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
        return await run_workflow_http(self._application, body, body.execution_context)


app = ValidationProstateHeatmapApplication.bind()  # type: ignore[attr-defined]
