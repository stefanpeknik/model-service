"""Today's model-binding fields, derived from a served model (transitional).

Builders verify a model by comparing its `get_config()` with the binding the
API's model catalog holds. Until "Retire the builders and transitional code"
removes both, a served model reports and is bound by these fields.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from model_service.catalog.served_models.fields import ScoresOutput


if TYPE_CHECKING:
    from model_service.catalog.served_models.declaration import ServedModel


def legacy_config(model: ServedModel) -> dict[str, Any]:
    """The fields today's `get_config()` returns and today's binding holds.

    Args:
        model: The served model.

    Returns:
        For scores: `tile_size`, `output_tile_size`, `n_channels`, `mpp`.
        For embeddings: `tile_size`, `mpp`, `summaries`.
    """
    common = {"tile_size": model.input.tile_size, "mpp": model.input.mpp}
    output = model.output
    if isinstance(output, ScoresOutput):
        return common | {"output_tile_size": output.size, "n_channels": output.channels}
    return common | {"summaries": list(output.summaries)}
