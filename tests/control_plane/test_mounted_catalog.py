"""The control plane's catalog, read from the directory the chart's ConfigMap mounts."""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from model_service.catalog import load_catalog
from model_service.control_plane.catalog_files import CatalogError
from model_service.control_plane.catalog_mount import MountedCatalog
from tests.support.model_documents import model_document
from tests.support.paths import RATIONAI_TEST


if TYPE_CHECKING:
    from collections.abc import Callable


CATALOG = Path(__file__).resolve().parents[2] / "catalog"
SMALL = model_document()


def heatmap() -> dict[str, Any]:
    return load_catalog(CATALOG).pipelines["prostate-heatmap"].declaration


def prostate() -> dict[str, Any]:
    entry = load_catalog(CATALOG, RATIONAI_TEST).models["prostate-classifier-1"]
    return entry.declaration


def mount(
    directory: Path,
    served: dict[str, Any],
    pipelines: dict[str, Any] | None = None,
) -> Path:
    """Write the file the chart's catalog ConfigMap holds."""
    directory.mkdir(exist_ok=True)
    content = {"models": served, "pipelines": pipelines or {}}
    (directory / "catalog.json").write_text(json.dumps(content), encoding="utf-8")
    return directory


def model_ids(catalog: MountedCatalog) -> list[str]:
    return list(catalog.resolved.models)


def test_a_changed_catalog_is_picked_up_in_place(tmp_path: Path) -> None:
    directory = mount(tmp_path / "mount", {"small": SMALL})
    catalog = MountedCatalog(directory)

    unchanged = catalog.refresh()
    mount(directory, {"small": SMALL, "prostate-classifier-1": prostate()})
    changed = catalog.refresh()

    assert (unchanged, changed) == (False, True)
    assert model_ids(catalog) == ["prostate-classifier-1", "small"]
    assert catalog.status().status == "loaded"


def test_an_invalid_catalog_keeps_the_old_one_and_reports_why(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    directory = mount(tmp_path / "mount", {"small": SMALL})
    catalog = MountedCatalog(directory)

    mount(directory, {"small": model_document(tile_size=0)})
    with caplog.at_level(logging.ERROR):
        catalog.refresh()
    failed = catalog.status()
    mount(directory, {"other": SMALL})
    catalog.refresh()

    assert failed.status == "reload_failed"
    assert failed.error is not None
    assert "catalog/models/small.yaml:input.tile_size" in failed.error
    assert "catalog/models/small.yaml:input.tile_size" in caplog.text
    assert model_ids(catalog) == ["other"]
    assert catalog.status().status == "loaded"


def test_a_catalog_that_does_not_load_at_startup_stops_the_control_plane(
    tmp_path: Path,
) -> None:
    directory = mount(tmp_path / "mount", {"Bad ID": SMALL})

    with pytest.raises(CatalogError, match="Bad ID"):
        MountedCatalog(directory)
    with pytest.raises(CatalogError):
        MountedCatalog(tmp_path / "missing")


def test_pipelines_load_beside_the_models_they_call(tmp_path: Path) -> None:
    directory = mount(
        tmp_path / "mount",
        {"prostate-classifier-1": prostate()},
        pipelines={"prostate-heatmap": heatmap()},
    )

    catalog = MountedCatalog(directory)

    (pipeline,) = catalog.resolved.pipelines.values()
    assert pipeline.id == "prostate-heatmap"
    assert list(pipeline.models) == ["prostate-classifier-1"]


def test_an_pipeline_naming_an_undeployed_model_names_it(
    tmp_path: Path,
) -> None:
    directory = mount(
        tmp_path / "mount", {"small": SMALL}, pipelines={"prostate-heatmap": heatmap()}
    )

    with pytest.raises(CatalogError, match="no model `prostate-classifier-1`"):
        MountedCatalog(directory)


def test_a_reload_brings_new_pipelines(tmp_path: Path) -> None:
    directory = mount(tmp_path / "mount", {"prostate-classifier-1": prostate()})
    catalog = MountedCatalog(directory)

    mount(
        directory,
        {"prostate-classifier-1": prostate()},
        pipelines={"prostate-heatmap": heatmap()},
    )
    catalog.refresh()

    assert list(catalog.resolved.pipelines) == ["prostate-heatmap"]


async def until(condition: Callable[[], bool]) -> None:
    for _ in range(500):
        if condition():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition never held")


@pytest.mark.anyio
async def test_the_watcher_survives_a_failing_reload_and_reports_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = mount(tmp_path / "mount", {"small": SMALL})
    catalog = MountedCatalog(directory)
    watcher = asyncio.create_task(catalog.watch(0.01))
    try:
        # The temporary catalog can't be written: an OSError, not an invalid catalog.
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "missing"))
        mount(directory, {"other": SMALL})
        await until(lambda: catalog.status().status == "reload_failed")
        failed = catalog.status()
        kept = model_ids(catalog)
        monkeypatch.setattr(tempfile, "tempdir", None)
        await until(lambda: catalog.status().status == "loaded")
    finally:
        watcher.cancel()

    assert failed.error
    assert kept == ["small"]
    assert model_ids(catalog) == ["other"]


@pytest.mark.anyio
async def test_a_stopped_watcher_is_reported(tmp_path: Path) -> None:
    catalog = MountedCatalog(mount(tmp_path / "mount", {"small": SMALL}))
    watcher = asyncio.create_task(catalog.watch(0.01))
    await asyncio.sleep(0.03)
    running = catalog.watcher_stopped

    watcher.cancel()
    await asyncio.gather(watcher, return_exceptions=True)

    assert (running, catalog.watcher_stopped) == (False, True)
