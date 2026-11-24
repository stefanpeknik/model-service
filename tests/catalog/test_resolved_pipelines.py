"""Resolving a pipeline for one run, and reading that document back on the runner."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from model_service.catalog import load_catalog
from model_service.catalog.pipelines import (
    ResolvedPipelineInvalid,
    load_resolved,
    resolve_run,
)
from model_service.contracts.failures import InvalidInvocation, UnknownPipelineError
from tests.support.paths import RATIONAI_TEST
from tests.support.pipeline_catalog import REGISTRY, model_file, write


if TYPE_CHECKING:
    from model_service.catalog import Catalog


CATALOG = Path(__file__).resolve().parents[2] / "catalog"


@pytest.fixture
def catalog(tmp_path: Path) -> Catalog:
    root = tmp_path / "catalog"
    write(root, "models/m1.yaml", model_file())
    write(
        root,
        "pipelines/a1.yaml",
        """
        summary: Scores tiles.
        source: {use: slide_tiles}
        steps: [{model: m1}]
        sinks: [{use: score_grid, output: heatmap}]
        """,
    )
    return load_catalog(root, registry=REGISTRY)


def test_a_run_resolves_every_option_and_names_its_models(
    catalog: Catalog,
) -> None:
    resolved = resolve_run(catalog, "a1", {"stride_fraction": 0.25})
    m1 = catalog.models["m1"]

    assert resolved.document["source"] == {
        "use": "slide_tiles",
        "tile_size": 64,
        "mpp": 0.5,
        "region_of_interest": "optional",
        "stride_fraction": 0.25,
        "skip_background": True,
        "min_tissue_fraction": 0.0,
    }
    assert resolved.document["steps"] == [{"model": "m1"}]
    assert resolved.document["sinks"] == [{"use": "score_grid", "output": "heatmap"}]
    assert resolved.document["models"] == {
        "m1": {"digest": m1.digest, "declaration": m1.declaration}
    }
    assert resolved.pipeline.settings()["stride_fraction"] == 0.25


def test_the_runner_reads_back_exactly_what_was_resolved(catalog: Catalog) -> None:
    resolved = resolve_run(catalog, "a1", {"skip_background": False})

    loaded = load_resolved(copy.deepcopy(resolved.document), registry=REGISTRY)

    assert loaded.document == resolved.document
    assert loaded.digest == resolved.digest
    assert loaded.pipeline.settings() == resolved.pipeline.settings()
    assert loaded.model_digests == {"m1": catalog.models["m1"].digest}


@pytest.mark.parametrize("environment", [None, RATIONAI_TEST])
def test_the_repo_pipelines_read_back(environment: Path | None) -> None:
    catalog = load_catalog(CATALOG, environment)
    for pipeline_id in catalog.pipelines:
        resolved = resolve_run(catalog, pipeline_id)

        assert load_resolved(resolved.document).document == resolved.document


def test_the_digest_follows_settings_and_models(
    catalog: Catalog, tmp_path: Path
) -> None:
    default = resolve_run(catalog, "a1")
    tuned = resolve_run(catalog, "a1", {"stride_fraction": 0.25})
    write(tmp_path / "catalog", "models/m1.yaml", model_file(mpp=0.25))
    other_model = resolve_run(
        load_catalog(tmp_path / "catalog", registry=REGISTRY), "a1"
    )

    assert len({default.digest, tuned.digest, other_model.digest}) == 3
    assert resolve_run(catalog, "a1").digest == default.digest


def test_a_changed_document_is_refused(catalog: Catalog) -> None:
    document = copy.deepcopy(resolve_run(catalog, "a1").document)
    document["source"]["stride_fraction"] = 0.1

    with pytest.raises(ResolvedPipelineInvalid, match="digest"):
        load_resolved(document, registry=REGISTRY)


def test_settings_the_pipeline_does_not_take_are_refused_by_name(
    catalog: Catalog,
) -> None:
    with pytest.raises(InvalidInvocation, match="stride"):
        resolve_run(catalog, "a1", {"stride": 0.25})
    with pytest.raises(InvalidInvocation, match="stride_fraction"):
        resolve_run(catalog, "a1", {"stride_fraction": -1})


def test_an_unknown_pipeline_is_its_own_error(catalog: Catalog) -> None:
    with pytest.raises(UnknownPipelineError):
        resolve_run(catalog, "missing")
