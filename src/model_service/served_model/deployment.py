"""How a served-model replica reads back the `user_config` Helm gives it.

The chart (`templates/_helpers.tpl`) renders each served model's Serve
application; `host_config` resolves its `user_config` without importing Ray.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from model_service.catalog.served_models import ServedModel, resolve_served_model


DEPLOYMENT_NAME = "ServedModel"
DEFAULT_CACHE_ROOT = "/mnt/cache/trt_cache"


def host_config(user_config: dict[str, Any]) -> tuple[ServedModel, Path]:
    """Read the host's `user_config` back into the served model it describes.

    Args:
        user_config: `id`, `declaration` (the served-model document after its
            environment's patch) and optionally `cache_root`.

    Returns:
        The served model and the cache root.

    Raises:
        CatalogInvalid: If the declaration does not resolve.
        KeyError: If `id` or `declaration` is missing.
    """
    model = resolve_served_model(user_config["id"], user_config["declaration"])
    return model, Path(user_config.get("cache_root", DEFAULT_CACHE_ROOT))
