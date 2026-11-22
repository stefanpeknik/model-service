"""What today's handle methods return, so builders call a hosted model unchanged.

Removed with the builders in "Retire the builders and transitional code".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_service.catalog.served_models import ScoresOutput


if TYPE_CHECKING:
    import numpy as np

    from model_service.catalog.served_models import ServedModel


def legacy_prediction(model: ServedModel, output: np.ndarray) -> float | np.ndarray:
    """Today's `predict` result for one tile's `infer` output.

    Args:
        model: The served model.
        output: What `infer` returned for the tile.

    Returns:
        A Python float for one score per tile, as the old binary classifier returned;
        the array otherwise.
    """
    declared = model.output
    if isinstance(declared, ScoresOutput) and declared.channels == declared.size == 1:
        return float(output.reshape(-1)[0])
    return output
