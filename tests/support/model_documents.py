"""Rendered model files, as the chart passes them to the API's catalog."""

from __future__ import annotations

from typing import Any


def spatial_document(model_id: str = "prostate-classifier-1") -> dict[str, Any]:
    return {
        "inference": {"contract": "spatial-scores"},
        "serve": {
            "name": model_id,
            "route_prefix": f"/{model_id}",
            "import_path": "model_service.inference.implementations.binary_classifier:app",
            "deployment": {
                "name": "BinaryClassifier",
                "user_config": {
                    "tile_size": 512,
                    "output_tile_size": 1,
                    "n_channels": 1,
                    "mpp": 0.468,
                    "max_batch_size": 64,
                    "model": {"_target_": "model_service.inference.artifacts:mlflow"},
                },
            },
        },
    }


def embedding_document(model_id: str = "virchow2") -> dict[str, Any]:
    return {
        "inference": {"contract": "tile-embedding"},
        "serve": {
            "name": model_id,
            "route_prefix": f"/{model_id}",
            "import_path": "model_service.inference.implementations.virchow2:app",
            "deployment": {
                "name": "Virchow2",
                "user_config": {
                    "tile_size": 224,
                    "mpp": 0.5,
                    "summaries": ["global", "local_mean"],
                },
            },
        },
    }
