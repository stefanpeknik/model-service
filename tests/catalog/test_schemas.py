"""The generated JSON Schemas accept the repo's catalog files and catch typos.

Editors read these schemas, so a valid file must show no error and a
misspelt option must show one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator

from model_service.reference.schemas import schema_files
from model_service.registry import default_registry


CATALOG = Path(__file__).resolve().parents[2] / "catalog"
SCHEMAS = {
    path.name: Draft202012Validator(json.loads(text))
    for path, text in schema_files(default_registry(), Path()).items()
}


def schema_of(path: Path) -> str:
    """The schema a catalog file names on its first line."""
    return path.read_text().splitlines()[0].rsplit("/", 1)[-1]


@pytest.mark.parametrize(
    "path",
    sorted(p for p in CATALOG.rglob("*.yaml") if ".schemas" not in p.parts),
    ids=lambda p: str(p.relative_to(CATALOG)),
)
def test_every_catalog_file_fits_the_schema_it_names(path: Path) -> None:
    validator = SCHEMAS[schema_of(path)]

    errors = [
        e.message for e in validator.iter_errors(yaml.safe_load(path.read_text()))
    ]

    assert errors == []


@pytest.mark.parametrize(
    ("schema", "document"),
    [
        (
            "served-models.schema.json",
            {
                "loader": {"use": "mlflow", "uri": "mlflow-artifacts:/1/a/model.onnx"},
                "runtime": {"use": "onnx", "thread": 4},
                "input": {"tile_size": 512, "mpp": 0.5},
                "output": {"kind": "scores"},
                "batching": {"max_size": 8, "wait_s": 0.1},
                "resources": {"memory_gib": 4},
            },
        ),
        ("served-models.patch.schema.json", {"runtime": {"thread": 4}}),
        (
            "analyses.schema.json",
            {
                "source": {"use": "slide_tiles", "stride": 0.5},
                "sinks": [{"use": "score_grid", "output": "heatmap"}],
            },
        ),
    ],
    ids=["served model", "patch", "analysis"],
)
def test_a_misspelt_option_is_flagged(schema: str, document: dict[str, Any]) -> None:
    assert list(SCHEMAS[schema].iter_errors(document)) != []
