"""A resolved catalog: every model and pipeline, checked."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Mapping

    from model_service.catalog.models import Model
    from model_service.catalog.pipelines import Pipeline


@dataclass(frozen=True)
class Catalog:
    """Every document of a catalog directory for one environment.

    Each resolved document keeps its `declaration`, the file after the
    environment's patch.

    Attributes:
        models: Models by ID.
        pipelines: Pipelines by ID.
    """

    models: Mapping[str, Model] = field(default_factory=dict)
    pipelines: Mapping[str, Pipeline] = field(default_factory=dict)
