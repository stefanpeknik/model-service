from workflow_execution.catalog import ROOT, WorkflowCatalog
from workflow_execution.prostate_heatmap import (
    default_configuration,
    validate_configuration,
)


def default_catalog() -> WorkflowCatalog:
    return WorkflowCatalog.from_directory(
        ROOT / "workflow-definitions",
        configuration_validators={"prostate-heatmap": validate_configuration},
        configuration_defaults={"prostate-heatmap": default_configuration()},
    )
