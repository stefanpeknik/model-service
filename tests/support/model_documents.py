"""Rendered model files, as the chart passes them to the facade's catalog."""

from __future__ import annotations


def spatial_document(model_id: str = "prostate-classifier-1") -> dict[str, object]:
    return {
        "inference": {"contract": "spatial-scores"},
        "serve": {
            "name": model_id,
            "route_prefix": f"/{model_id}",
            "import_path": "model_service.models.binary_classifier:app",
            "deployments": [
                {
                    "name": "BinaryClassifier",
                    "user_config": {
                        "tile_size": 512,
                        "output_tile_size": 1,
                        "n_channels": 1,
                        "mpp": 0.468,
                        "max_batch_size": 64,
                        "model": {"_target_": "model_service.models.artifacts:mlflow"},
                    },
                }
            ],
        },
    }


def embedding_document(model_id: str = "virchow2") -> dict[str, object]:
    return {
        "inference": {
            "contract": "tile-embedding",
            "mpp": 0.5,
        },
        "serve": {
            "name": model_id,
            "route_prefix": f"/{model_id}",
            "import_path": "model_service.models.virchow2:app",
            "deployments": [
                {"name": "Virchow2", "user_config": {"tile_size": 224}},
            ],
        },
    }
