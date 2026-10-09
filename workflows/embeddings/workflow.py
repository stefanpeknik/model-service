"""Own one embedding invocation's HTTP client and atomic output publication."""

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
    """Convert provider-neutral slide/mask paths into a completed feature artifact."""

    def __init__(
        self,
        model: EmbeddingModel,
        settings: WorkflowWorkerSettings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Retain deployment wiring without opening files or making model requests.

        Args:
            model: Raw-model binding and expected summary order.
            settings: Thread/concurrency limits and internal Ray Serve origin.
            transport: Optional external HTTP transport, used by offline validation.
        """
        self._model = model
        self._settings = settings
        self._transport = transport

    async def run(self, context: WorkflowContext) -> WorkflowResult:
        """Compute an HDF5 artifact with invocation-owned resources and atomic completion.

        Args:
            context: Validated input access, resolved settings, and absolute deadline.

        Returns:
            WorkflowResult: The completed embeddings artifact with HDF5 media type.

        Raises:
            InvalidWorkflowInvocation: Settings, paths, or output/input aliasing are invalid.
            WorkflowFailure: Raw inference fails or returns invalid features.
            OSError: Input/output access or artifact publication fails.
            asyncio.CancelledError: Cancellation removes the temporary output and joins work.
        """
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
