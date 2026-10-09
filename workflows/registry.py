"""Register built-in workflow definitions with their configuration validators and defaults."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from workflow_execution.catalog import WorkflowCatalog, WorkflowConfiguration
from workflows.embeddings.configuration import EmbeddingConfiguration
from workflows.embeddings.configuration import (
    validate_configuration as validate_embeddings,
)
from workflows.heatmap.prostate import (
    default_configuration,
    validate_configuration,
)
from workflows.segmentation.episeg import (
    SegmentationConfiguration,
)
from workflows.segmentation.episeg import (
    validate_configuration as validate_segmentation,
)


if TYPE_CHECKING:
    import httpx

    from workflow_execution.application import Workflow
    from workflows.settings import WorkflowWorkerSettings


def default_catalog() -> WorkflowCatalog:
    """Load built-in YAML contracts with their complete configuration registrations.

    Returns:
        WorkflowCatalog: Indexed workflow contracts and configuration registrations.
    """
    return WorkflowCatalog.from_directory(
        Path(__file__).resolve().parents[1] / "workflow-definitions",
        configurations={
            "prostate-heatmap": WorkflowConfiguration(
                validator=validate_configuration,
                defaults=default_configuration(),
            ),
            "episeg-segmentation": WorkflowConfiguration(
                validator=validate_segmentation,
                defaults=SegmentationConfiguration().as_values(),
            ),
            "gigapath-embeddings": WorkflowConfiguration(
                validator=validate_embeddings,
                defaults=EmbeddingConfiguration().model_dump(),
            ),
            "virchow2-embeddings": WorkflowConfiguration(
                validator=validate_embeddings,
                defaults=EmbeddingConfiguration().model_dump(),
            ),
        },
    )


def build_workflow(
    workflow_id: str,
    settings: WorkflowWorkerSettings,
    *,
    model_transport: httpx.AsyncBaseTransport | None = None,
) -> Workflow:
    """Construct a new standalone workflow with deployment-only model wiring.

    Args:
        workflow_id: Identifier of the segmentation or embedding workflow to construct.
        settings: Worker limits and internal raw-model HTTP origin.
        model_transport: Optional external model transport for offline validation.

    Returns:
        Workflow: Computation implementation wrapped by the generic workflow runtime.

    Raises:
        KeyError: The requested identifier has no construction registration.
    """
    from workflows.heatmap.processor import SlideHeatmapProcessor
    from workflows.heatmap.ray_predictor import RayHeatmapPredictor
    from workflows.segmentation.episeg import EPISegWorkflow

    if workflow_id == "episeg-segmentation":
        return EPISegWorkflow(
            SlideHeatmapProcessor(
                RayHeatmapPredictor("episeg-1"),
                num_threads=settings.num_threads,
                max_concurrent_tasks=settings.max_concurrent_tasks,
            )
        )
    if workflow_id in _EMBEDDING_MODELS:
        from workflows.embeddings.predictor import EmbeddingModel
        from workflows.embeddings.workflow import EmbeddingWorkflow

        model_id, summaries = _EMBEDDING_MODELS[workflow_id]
        return EmbeddingWorkflow(
            EmbeddingModel(model_id, summaries), settings, transport=model_transport
        )
    raise KeyError(workflow_id)


_EMBEDDING_MODELS = {
    "gigapath-embeddings": ("prov-gigapath", ("tile",)),
    "virchow2-embeddings": ("virchow2", ("global", "local_mean")),
}
