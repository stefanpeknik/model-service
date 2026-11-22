"""Model files as the chart passes them to the API's catalog: the file plus its `id`."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

import yaml


if TYPE_CHECKING:
    from pathlib import Path
    from types import ModuleType


IMPLEMENTATIONS = "model_service.inference.implementations"


def spatial_document(model_id: str = "prostate-classifier-1") -> dict[str, Any]:
    return {
        "id": model_id,
        "contract": "spatial-scores",
        "implementation": "binary_classifier.BinaryClassifier",
        "deployment": {
            "user_config": {
                "tile_size": 512,
                "output_tile_size": 1,
                "n_channels": 1,
                "mpp": 0.468,
                "max_batch_size": 64,
                "model": {"_target_": "model_service.inference.artifacts:mlflow"},
            },
        },
    }


def embedding_document(model_id: str = "virchow2") -> dict[str, Any]:
    return {
        "id": model_id,
        "contract": "tile-embedding",
        "implementation": "virchow2.Virchow2",
        "deployment": {
            "user_config": {
                "tile_size": 224,
                "mpp": 0.5,
                "summaries": ["global", "local_mean"],
            },
        },
    }


def served_document(tile_size: int = 64) -> dict[str, Any]:
    """A small served-model document, as the chart puts it in `catalog.json`."""
    return {
        "weights": {"use": "local", "path": "/weights/model.onnx"},
        "runtime": {"use": "onnx"},
        "input": {"tile_size": tile_size, "mpp": 0.5},
        "output": {"kind": "embedding", "summaries": ["global"]},
        "batching": {"max_size": 8, "wait_s": 0},
        "resources": {"memory_gib": 1},
    }


def model_file(path: Path) -> dict[str, Any]:
    """Read a model file as Helm renders it into the catalog: its name is its `id`."""
    document: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {**document, "id": path.stem}


def implementation(document: dict[str, Any]) -> tuple[ModuleType, Any]:
    """Import a model file's `implementation` the way the chart names it to Ray.

    Returns:
        The module, which Ray loads as `<module>:app`, and its deployment class.
    """
    module_name, _, class_name = document["implementation"].partition(".")
    module = importlib.import_module(f"{IMPLEMENTATIONS}.{module_name}")
    return module, getattr(module, class_name)
