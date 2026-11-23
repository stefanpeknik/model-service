"""What a source or sink is told about the run it takes part in."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from model_service.catalog.served_models.declaration import ServedModel
    from model_service.components.columns import Col


@dataclass(frozen=True)
class Run:
    """The run context sources and sinks receive.

    Attributes:
        work_dir: Where a sink writes its artifact. Each sink gets its own
            directory; the runner moves the artifact to its destination only
            when the whole run succeeded.
        model: The served-model declaration of the analysis's last model step,
            or None for a model-free analysis.
        columns: The columns that reach the sinks, with every dimension the
            catalog knows filled in, e.g. `scores: float32[1,1,1]`.
    """

    work_dir: Path
    model: ServedModel | None = None
    columns: Mapping[str, Col] = field(default_factory=dict)
