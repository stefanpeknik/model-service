"""The Serve application a served-model file deploys as.

`serve_application` is the reference for what Helm renders per served model;
`host_config` is how the host reads its `user_config` back. Neither imports
Ray, so the chart's tests and the API can use them.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from model_service.catalog.served_models import ServedModel, resolve_served_model


if TYPE_CHECKING:
    from model_service.catalog import Entry


IMPORT_PATH = "model_service.served_model.serve:app"
DEPLOYMENT_NAME = "ServedModel"
DEFAULT_CACHE_ROOT = "/mnt/cache/trt_cache"
GIB = 1024**3


def serve_application(entry: Entry, *, cache_root: str) -> dict[str, Any]:
    """One `serveConfigV2` application for a served model.

    Helm adds `ray_actor_options.runtime_env` (e.g. `MLFLOW_TRACKING_URI`)
    from the environment's values; everything else is here.

    Args:
        entry: The served model's catalog entry.
        cache_root: The environment's cache root, on a shared volume.

    Returns:
        The application: name and route prefix are the model ID, the import
        path is the same for every served model, and its one deployment
        carries the Serve options and the `user_config`.
    """
    model: ServedModel = entry.value
    return {
        "name": model.id,
        "route_prefix": f"/{model.id}",
        "import_path": IMPORT_PATH,
        "deployments": [
            {
                "name": DEPLOYMENT_NAME,
                **deployment_options(model),
                "user_config": {
                    "id": model.id,
                    "declaration": entry.document,
                    "cache_root": cache_root,
                },
            }
        ],
    }


def deployment_options(model: ServedModel) -> dict[str, Any]:
    """The Serve deployment options from `resources`, `batching` and `serving`.

    A `serving` limit left out follows from `batching.max_size`: the
    autoscaler aims at half a batch per replica, a replica takes two batches
    at once and sixteen wait in its queue. `downscale_delay_s` is passed
    only when given.

    Args:
        model: The served model.

    Returns:
        `max_ongoing_requests`, `max_queued_requests`, `autoscaling_config` and
        `ray_actor_options` (CPUs, GPUs and memory in bytes).
    """
    batch = model.batching.max_size
    serving = model.serving
    autoscaling: dict[str, Any] = {
        "min_replicas": serving.min_replicas,
        "max_replicas": serving.max_replicas,
        "target_ongoing_requests": serving.target_ongoing_requests
        or max(batch // 2, 1),
    }
    if serving.downscale_delay_s is not None:
        autoscaling["downscale_delay_s"] = serving.downscale_delay_s
    return {
        "max_ongoing_requests": serving.max_ongoing_requests or 2 * batch,
        "max_queued_requests": serving.max_queued_requests or 16 * batch,
        "autoscaling_config": autoscaling,
        "ray_actor_options": {
            "num_cpus": model.resources.cpus,
            "num_gpus": model.resources.gpus,
            "memory": int(model.resources.memory_gib * GIB),
        },
    }


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
