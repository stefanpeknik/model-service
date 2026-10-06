from pathlib import Path

from workflow_execution.catalog import WorkflowCatalog, WorkflowConfiguration
from workflows.heatmap.prostate import (
    default_configuration,
    validate_configuration,
)


def default_catalog() -> WorkflowCatalog:
    return WorkflowCatalog.from_directory(
        Path(__file__).resolve().parents[1] / "workflow-definitions",
        configurations={
            "prostate-heatmap": WorkflowConfiguration(
                validator=validate_configuration,
                defaults=default_configuration(),
            )
        },
    )
