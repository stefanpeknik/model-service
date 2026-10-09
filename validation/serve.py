"""Host the historical dummy heatmap workflow used for deployment plumbing checks.

This processor writes an 8-by-8 synthetic TIFF, not a native whole-slide heatmap. It can
delay cancellation scenarios and optionally exercise mock EMPAIA input access.
"""

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
    """Produce a fixed synthetic TIFF to exercise workflow I/O and cancellation without model inference."""

    async def build(self, request: HeatmapBuildRequest) -> None:
        """Read both fixture files, optionally pause a cancellation scenario, and write the synthetic 8-by-8 TIFF.

        Args:
            request: Slide/mask files, destination, stride fraction, and TIFF tile
                geometry.

        Returns:
            None: No return value.
        """
        request.slide_path.read_bytes()
        request.tissue_mask_path.read_bytes()
        if "cancellation" in request.output_path.parts:
            await asyncio.sleep(300)
        Image.new("L", (8, 8), color=128).save(request.output_path, format="TIFF")


fastapi = FastAPI()


@serve.deployment(num_replicas="auto")
@serve.ingress(fastapi)
class ValidationProstateHeatmapApplication:
    """Expose the dummy processor through the same workflow contract as the native application."""

    def __init__(self) -> None:
        """Wire the dummy workflow wrapper and optionally register EMPAIA workspace access from environment settings."""
        catalog = WorkflowCatalog.default()
        definition = catalog.get("prostate-heatmap")
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
        """Execute the internal workflow request through the shared safe HTTP adapter.

        Args:
            body: Internal invocation envelope including private processing attempt
                context.

        Returns:
            WorkflowResult: Named inline outputs and/or references to completed artifact
                files.
        """
        return await run_workflow_http(self._application, body, body.execution_context)


app = ValidationProstateHeatmapApplication.bind()  # type: ignore[attr-defined]
