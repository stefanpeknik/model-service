from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from PIL import Image
from ray import serve

from workflow_execution.application import WorkflowApplication
from workflow_execution.catalog import WorkflowCatalog
from workflow_execution.contracts import WorkflowInvocation, WorkflowResult
from workflow_execution.prostate_heatmap import (
    HeatmapBuildRequest,
    ProstateHeatmapWorkflow,
)


class ValidationHeatmapProcessor:
    async def build(self, request: HeatmapBuildRequest) -> None:
        request.slide_path.read_bytes()
        request.tissue_mask_path.read_bytes()
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

    @fastapi.post("/", response_model=WorkflowResult)
    async def run(self, body: WorkflowInvocation) -> WorkflowResult:
        return await self._application.run(body)


app = ValidationProstateHeatmapApplication.bind()  # type: ignore[attr-defined]
