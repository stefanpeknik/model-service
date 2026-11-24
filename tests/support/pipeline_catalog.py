"""Small catalogs of models and pipelines, written into a test's directory.

`REGISTRY` holds this repo's components plus the test-only ones, so pipelines
can mix `slide_tiles` with test sinks and steps.
"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

from model_service.registry import Registry, default_registry


if TYPE_CHECKING:
    from pathlib import Path


REGISTRY = Registry(
    {
        (c.kind, c.name): c
        for c in [*default_registry(), *Registry.scan("tests.support.components")]
    }
)


def model_file(
    tile_size: int = 64, mpp: float = 0.5, output: str = "kind: scores"
) -> str:
    """A model file that resolves without weights on disk."""
    return f"""
        loader: {{use: local, path: /weights/model.onnx}}
        runtime: {{use: onnx}}
        input: {{tile_size: {tile_size}, mpp: {mpp}}}
        output: {{{output}}}
        batching: {{max_size: 8, wait_s: 0.01}}
        resources: {{memory_gib: 2}}
    """


def write(root: Path, name: str, text: str) -> None:
    """Write a catalog file, dedented, creating its folder."""
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))
