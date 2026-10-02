from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Mapping

    from workflow_execution.catalog import WorkflowCatalog
    from workflow_execution.contracts import JsonValue


@dataclass(frozen=True, slots=True)
class EmpaiaConfigurationRegistry:
    overrides: Mapping[str, Mapping[str, JsonValue]]

    @classmethod
    def default(cls) -> EmpaiaConfigurationRegistry:
        return cls({"prostate-heatmap": {"output_path": "heatmap.tiff"}})

    def resolve(
        self,
        catalog: WorkflowCatalog,
        workflow_id: str,
        configuration: Mapping[str, JsonValue],
    ) -> dict[str, JsonValue]:
        return catalog.resolve_configuration(
            workflow_id,
            configuration,
            overrides=self.overrides.get(workflow_id),
        )
