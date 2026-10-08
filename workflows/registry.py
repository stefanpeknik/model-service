"""Register built-in workflow definitions with their configuration validators and defaults."""

from pathlib import Path

from workflow_execution.catalog import WorkflowCatalog, WorkflowConfiguration
from workflows.heatmap.prostate import (
    default_configuration,
    validate_configuration,
)


def default_catalog() -> WorkflowCatalog:
    """Load repository YAML definitions and register the prostate heatmap validator with its complete defaults.

    Returns:
        WorkflowCatalog: Indexed workflow contracts and configuration registrations.
    """
    return WorkflowCatalog.from_directory(
        Path(__file__).resolve().parents[1] / "workflow-definitions",
        configurations={
            "prostate-heatmap": WorkflowConfiguration(
                validator=validate_configuration,
                defaults=default_configuration(),
            )
        },
    )
