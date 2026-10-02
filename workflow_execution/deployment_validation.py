from __future__ import annotations

from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from workflow_execution.catalog import MissingWorkflowApplicationsError


if TYPE_CHECKING:
    from pathlib import Path

    from workflow_execution.catalog import WorkflowCatalog


class _HelmValues(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")
    applications: list[str]


class _Application(BaseModel):
    model_config = ConfigDict(strict=True, extra="ignore")
    name: str = Field(min_length=1)


_APPLICATIONS = TypeAdapter(list[_Application])


def validate_helm_applications(
    catalog: WorkflowCatalog, values_path: Path, applications_directory: Path
) -> None:
    values = _HelmValues.model_validate(_load_yaml(values_path))
    catalog.validate_deployed(set(values.applications))
    missing = []
    for workflow_id in sorted(catalog.identifiers):
        try:
            applications = _APPLICATIONS.validate_python(
                _load_yaml(applications_directory / f"{workflow_id}.yaml")
            )
        except ValueError:
            missing.append(workflow_id)
            continue
        if not applications or applications[0].name != workflow_id:
            missing.append(workflow_id)
    if missing:
        raise MissingWorkflowApplicationsError(tuple(missing))


def _load_yaml(path: Path) -> object:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"cannot load YAML document {path}") from error
