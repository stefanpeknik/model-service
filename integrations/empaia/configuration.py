from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

from workflow_execution.prostate_heatmap import ProstateConfiguration


if TYPE_CHECKING:
    from collections.abc import Mapping

    from workflow_execution.contracts import JsonValue


@dataclass(frozen=True, slots=True)
class EmpaiaConfigurationRegistry:
    defaults: Mapping[str, Mapping[str, JsonValue]]

    @classmethod
    def default(cls) -> EmpaiaConfigurationRegistry:
        prostate = ProstateConfiguration(
            output_path="heatmap.tiff",
            stride_fraction=0.5,
            output_tile_height=256,
            output_tile_width=256,
        )
        return cls({"prostate-heatmap": prostate.as_values()})

    def resolve(
        self,
        workflow_id: str,
        configuration: Mapping[str, JsonValue],
    ) -> dict[str, JsonValue]:
        return deepcopy(dict(self.defaults.get(workflow_id, {}))) | deepcopy(
            dict(configuration)
        )
