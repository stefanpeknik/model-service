"""The heatmap builder's names for the `score_grid` format (transitional).

The layout lives in `model_service.formats.score_grid`. The builder stores it
as `application/x-npz`, and the EMPAIA publishers key on that, until "Retire
the builders and transitional code" removes this module.
"""

from __future__ import annotations

from model_service.formats.score_grid import (
    LEGACY_MEDIA_TYPE as MEDIA_TYPE,
)
from model_service.formats.score_grid import (
    ScoreGrid,
)
from model_service.formats.score_grid import (
    read as read_score_grid,
)
from model_service.formats.score_grid import (
    write as write_score_grid,
)


__all__ = ["MEDIA_TYPE", "ScoreGrid", "read_score_grid", "write_score_grid"]
