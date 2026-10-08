from __future__ import annotations

from typing import TYPE_CHECKING

import httpx

from workflow_execution.contracts import SharedArtifact, WorkflowResult
from workflows.artifacts import atomic_artifact
from workflows.embeddings.configuration import validate_configuration
from workflows.embeddings.predictor import HttpEmbeddingPredictor
from workflows.embeddings.processor import build_embeddings


if TYPE_CHECKING:
    from workflow_execution.application import WorkflowContext
    from workflows.embeddings.predictor import EmbeddingModel
    from workflows.settings import WorkflowWorkerSettings


class EmbeddingWorkflow:
    def __init__(
        self,
        model: EmbeddingModel,
        settings: WorkflowWorkerSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._model = model
        self._settings = settings
        self._transport = transport

    async def run(self, context: WorkflowContext) -> WorkflowResult:
        config = validate_configuration(context.configuration)
        slide = context.input_path("slide")
        mask = context.input_path("region_of_interest")
        destination = context.output_path(config.output_path)
        with atomic_artifact(destination, inputs=(slide, mask)) as temporary:
            async with httpx.AsyncClient(
                base_url=str(self._settings.model_origin),
                transport=self._transport,
                timeout=None,
                trust_env=False,
            ) as client:
                await build_embeddings(
                    slide,
                    mask,
                    temporary,
                    predictor=HttpEmbeddingPredictor(client, self._model),
                    model=self._model,
                    stride_fraction=config.stride_fraction,
                    settings=self._settings,
                )
        return WorkflowResult(
            outputs={
                "embeddings": SharedArtifact(
                    path=str(destination),
                    media_type="application/x-hdf5",
                )
            }
        )
