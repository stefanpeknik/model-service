"""Served-model documents as the chart puts them in the API's `catalog.json`."""

from __future__ import annotations

from typing import Any


def served_document(tile_size: int = 64) -> dict[str, Any]:
    """A small served-model document that resolves without weights on disk."""
    return {
        "weights": {"use": "local", "path": "/weights/model.onnx"},
        "runtime": {"use": "onnx"},
        "input": {"tile_size": tile_size, "mpp": 0.5},
        "output": {"kind": "embedding", "summaries": ["global"]},
        "batching": {"max_size": 8, "wait_s": 0},
        "resources": {"memory_gib": 1},
    }
