"""A resolved catalog: every served model and analysis, checked."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Mapping

    from model_service.catalog.analyses import Analysis
    from model_service.catalog.served_models import ServedModel


@dataclass(frozen=True)
class Catalog:
    """Every document of a catalog directory for one environment.

    Each resolved document keeps its `declaration`, the file after the
    environment's patch.

    Attributes:
        environment: The environment whose patches were applied, or None.
        served_models: Served models by ID.
        analyses: Analyses by ID.
    """

    environment: str | None
    served_models: Mapping[str, ServedModel] = field(default_factory=dict)
    analyses: Mapping[str, Analysis] = field(default_factory=dict)
