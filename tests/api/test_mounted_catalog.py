"""The API's catalog, read from the directory the chart's ConfigMap mounts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from model_service.api.catalog_mount import MountedCatalog
from model_service.api.model_catalog import CatalogError
from model_service.catalog import load_catalog
from model_service.contracts.model_declarations import (
    SpatialScoresModel,
    TileEmbeddingModel,
)
from tests.support.model_documents import (
    embedding_document,
    served_document,
    spatial_document,
)


CATALOG = Path(__file__).resolve().parents[2] / "catalog"
SMALL = served_document()


def prostate() -> dict[str, Any]:
    entry = load_catalog(CATALOG, "rationai-test").entries("served-models")[
        "prostate-classifier-1"
    ]
    return entry.document


def mount(
    directory: Path,
    served: dict[str, Any],
    legacy: list[dict[str, Any]] | None = None,
    blocks: dict[str, Any] | None = None,
) -> Path:
    """Write the files the chart's catalog ConfigMap holds."""
    directory.mkdir(exist_ok=True)
    files = {
        "catalog.json": {"served-models": served},
        "model-catalog.json": legacy or [],
        "provider-blocks.json": blocks or {},
    }
    for name, content in files.items():
        (directory / name).write_text(json.dumps(content), encoding="utf-8")
    return directory


def test_served_models_give_todays_bindings_beside_old_model_files(
    tmp_path: Path,
) -> None:
    directory = mount(
        tmp_path / "mount",
        {"prostate-classifier-1": prostate(), "small": SMALL},
        legacy=[embedding_document()],
        blocks={"prostate-classifier-1": {"empaia": {"name": "Prostate cancer"}}},
    )

    catalog = MountedCatalog(directory, provider_blocks=["empaia"])

    assert list(catalog) == [
        SpatialScoresModel(
            id="prostate-classifier-1",
            tile_size=512,
            output_tile_size=1,
            n_channels=1,
            mpp=0.468,
        ),
        TileEmbeddingModel(id="small", tile_size=64, mpp=0.5, summaries=("global",)),
        TileEmbeddingModel(
            id="virchow2", tile_size=224, mpp=0.5, summaries=("global", "local_mean")
        ),
    ]
    assert catalog.provider_block("empaia") == {
        "prostate-classifier-1": {"name": "Prostate cancer"}
    }


def test_an_old_model_file_repeating_a_served_model_is_left_out(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    old = spatial_document()
    old["deployment"]["user_config"]["mpp"] = 1.0
    directory = mount(
        tmp_path / "mount", {"prostate-classifier-1": prostate()}, legacy=[old]
    )

    with caplog.at_level(logging.ERROR):
        catalog = MountedCatalog(directory)

    assert [(m.id, m.mpp) for m in catalog] == [("prostate-classifier-1", 0.468)]
    assert "'prostate-classifier-1'" in caplog.text


def test_a_provider_block_no_provider_takes_is_refused(tmp_path: Path) -> None:
    directory = mount(
        tmp_path / "mount", {"small": SMALL}, blocks={"small": {"listing": {}}}
    )

    with pytest.raises(CatalogError, match="listing"):
        MountedCatalog(directory, provider_blocks=["empaia"])


def test_a_changed_catalog_is_picked_up_in_place(tmp_path: Path) -> None:
    directory = mount(tmp_path / "mount", {"small": SMALL})
    catalog = MountedCatalog(directory)

    unchanged = catalog.refresh()
    mount(directory, {"small": SMALL, "prostate-classifier-1": prostate()})
    changed = catalog.refresh()

    assert (unchanged, changed) == (False, True)
    assert [m.id for m in catalog] == ["prostate-classifier-1", "small"]
    assert catalog.status().status == "loaded"


def test_an_invalid_catalog_keeps_the_old_one_and_reports_why(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    directory = mount(tmp_path / "mount", {"small": SMALL})
    catalog = MountedCatalog(directory)

    mount(directory, {"small": served_document(tile_size=0)})
    with caplog.at_level(logging.ERROR):
        catalog.refresh()
    failed = catalog.status()
    mount(directory, {"other": SMALL})
    catalog.refresh()

    assert failed.status == "reload_failed"
    assert failed.error is not None
    assert "catalog/served-models/small.yaml:input.tile_size" in failed.error
    assert "catalog/served-models/small.yaml:input.tile_size" in caplog.text
    assert [m.id for m in catalog] == ["other"]
    assert catalog.status().status == "loaded"


def test_a_catalog_that_does_not_load_at_startup_stops_the_api(
    tmp_path: Path,
) -> None:
    directory = mount(tmp_path / "mount", {"Bad ID": SMALL})

    with pytest.raises(CatalogError, match="Bad ID"):
        MountedCatalog(directory)
    with pytest.raises(CatalogError):
        MountedCatalog(tmp_path / "missing")
