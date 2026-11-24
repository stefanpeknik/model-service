"""Pipeline files: what they resolve to and every load-time check."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from components import Col
from model_service.catalog import CatalogInvalid, load_catalog
from tests.support.paths import RATIONAI_TEST
from tests.support.pipeline_catalog import REGISTRY, model_file, write


if TYPE_CHECKING:
    from model_service.catalog.pipelines import Pipeline


CATALOG = Path(__file__).resolve().parents[2] / "catalog"

HEATMAP = """
    source: {use: slide_tiles}
    steps: [{model: m1}]
    sinks: [{use: score_grid, output: heatmap}]
"""


@pytest.fixture
def root(tmp_path: Path) -> Path:
    catalog = tmp_path / "catalog"
    write(catalog, "models/m1.yaml", model_file())
    return catalog


def pipeline(root: Path, pipeline_id: str = "a1") -> Pipeline:
    return load_catalog(root, registry=REGISTRY).pipelines[pipeline_id]


def errors(root: Path) -> list[str]:
    with pytest.raises(CatalogInvalid) as raised:
        load_catalog(root, registry=REGISTRY)
    return [str(error) for error in raised.value.errors]


def heatmap_ids() -> list[str]:
    pipelines = load_catalog(CATALOG).pipelines
    return sorted(
        i for i, e in pipelines.items() if any(s.name == "score_grid" for s in e.sinks)
    )


@pytest.mark.parametrize("environment", [None, RATIONAI_TEST])
@pytest.mark.parametrize("pipeline_id", heatmap_ids())
def test_heatmap_pipelines_tile_at_their_models_geometry(
    environment: Path | None, pipeline_id: str
) -> None:
    catalog = load_catalog(CATALOG, environment)
    heatmap = catalog.pipelines[pipeline_id]
    (model_id,) = heatmap.models
    served = catalog.models[model_id]

    assert heatmap.source.name == "slide_tiles"
    assert (heatmap.source.options.tile_size, heatmap.source.options.mpp) == (
        served.input.tile_size,
        served.input.mpp,
    )
    output = served.output
    assert heatmap.columns["scores"] == Col(
        "float32", (output.channels, output.size, output.size)
    )


@pytest.mark.parametrize("environment", [None, RATIONAI_TEST])
@pytest.mark.parametrize(
    ("pipeline_id", "score_map"),
    [("prostate-heatmap", (1, 1, 1)), ("episeg-heatmap", (1, 1024, 1024))],
)
def test_the_prostate_and_episeg_heatmaps_tile_half_a_tile_apart(
    environment: Path | None, pipeline_id: str, score_map: tuple[int, int, int]
) -> None:
    heatmap = load_catalog(CATALOG, environment).pipelines[pipeline_id]

    assert heatmap.settings() == {
        "stride_fraction": 0.5,
        "skip_background": True,
        "min_tissue_fraction": 0.0,
    }
    assert heatmap.inputs == {"slide": True, "region_of_interest": False}
    assert heatmap.columns["scores"] == Col("float32", score_map)
    assert [(s.name, s.options.output) for s in heatmap.sinks] == [
        ("score_grid", "heatmap")
    ]


@pytest.mark.parametrize(
    ("roi", "inputs"),
    [
        ("required", {"slide": True, "region_of_interest": True}),
        ("optional", {"slide": True, "region_of_interest": False}),
        ("ignored", {"slide": True}),
    ],
)
def test_inputs_follow_from_the_source_options(
    root: Path, roi: str, inputs: dict[str, bool]
) -> None:
    write(
        root,
        "pipelines/a1.yaml",
        HEATMAP.replace(
            "{use: slide_tiles}", f"{{use: slide_tiles, region_of_interest: {roi}}}"
        ),
    )

    assert pipeline(root).inputs == inputs


def test_a_caller_overrides_per_run_settings_and_nothing_else(root: Path) -> None:
    write(root, "pipelines/a1.yaml", HEATMAP)
    resolved = pipeline(root)

    tuned = resolved.with_settings({"stride_fraction": 1.0})

    assert tuned.settings()["stride_fraction"] == 1.0
    assert resolved.settings()["stride_fraction"] == 0.5
    with pytest.raises(ValueError, match="settings: min_tissue_fraction, "):
        resolved.with_settings({"tile_size": 128})
    with pytest.raises(ValueError, match="stride_fraction"):
        resolved.with_settings({"stride_fraction": 2})


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        pytest.param(
            {"pipelines/a1.yaml": HEATMAP.replace("model: m1", "model: m2")},
            [
                "catalog/pipelines/a1.yaml:steps[0].model: no model `m2`. "
                "Models: m1. Hint: add catalog/models/m2.yaml or "
                "fix the name"
            ],
            id="unknown model",
        ),
        pytest.param(
            {
                "pipelines/a1.yaml": HEATMAP.replace(
                    "{use: slide_tiles}", "{use: slide_tiles, tile_size: 128}"
                )
            },
            [
                "catalog/pipelines/a1.yaml:source.tile_size: `128`, but model step "
                "`m1` takes 64. The first model step's value. Hint: leave "
                "`tile_size` and `mpp` out so the first model step sets them; a "
                "model-free pipeline sets both"
            ],
            id="geometry conflicts with the model",
        ),
        pytest.param(
            {
                "models/m2.yaml": model_file(mpp=1.0),
                "pipelines/a1.yaml": HEATMAP.replace(
                    "[{model: m1}]", "[{model: m1}, {model: m2}]"
                ),
            },
            [
                "catalog/pipelines/a1.yaml:steps[1].model: model step `m2` takes 64 "
                "px tiles at 1.0 µm/px, but the first model step `m1` takes 64 px "
                "at 0.5 µm/px. Model steps that read the same tiles. Hint: a "
                "pipeline reads one tile geometry; split it into two pipelines"
            ],
            id="two model steps with different geometry",
        ),
        pytest.param(
            {
                "pipelines/a1.yaml": """
                    source: {use: slide_tiles}
                    sinks: [{use: score_grid, output: heatmap}]
                """
            },
            [
                "catalog/pipelines/a1.yaml:source: a model-free pipeline must set "
                "`tile_size` and `mpp`. E.g. `source: {use: slide_tiles, "
                "tile_size: 256, mpp: 1.0}`. Hint: leave `tile_size` and `mpp` "
                "out so the first model step sets them; a model-free pipeline "
                "sets both",
                "catalog/pipelines/a1.yaml:sinks[0]: `score_grid` reads `scores`, "
                "which nothing before it adds. Added by: a model step with "
                "`output: {kind: scores}`, mean_intensity. Hint: put a component "
                "that adds `scores` before it",
            ],
            id="model-free without geometry or scores",
        ),
        pytest.param(
            {
                "models/m1.yaml": model_file(output="kind: scores, size: 4"),
                "pipelines/a1.yaml": HEATMAP.replace(
                    "use: score_grid, output: heatmap", "use: score_table, output: t"
                ),
            },
            [
                "catalog/pipelines/a1.yaml:sinks[0]: `score_table` reads `scores` "
                "as float32[?,1,1], but model step `m1` adds float32[1,4,4] "
                "(model m1). A column float32[?,1,1]. Hint: pick a model or "
                "step whose `scores` fits, or a sink that reads float32[1,4,4]",
            ],
            id="column shape mismatch",
        ),
        pytest.param(
            {
                "pipelines/a1.yaml": """
                    source: {use: grid, tile_size: 64}
                    steps: [{model: m1}]
                    sinks: [{use: score_grid, output: heatmap}]
                """
            },
            [
                "catalog/pipelines/a1.yaml:sinks[0]: `score_grid` reads frame "
                "fields level, mpp_x, mpp_y, stride_x, stride_y, which `grid` "
                "doesn't give. Frame fields of `grid`: extent_x, extent_y, "
                "tile_extent_x, tile_extent_y. Hint: use a source that gives "
                "them, e.g. slide_tiles",
                "catalog/pipelines/a1.yaml:sinks[0]: `score_grid` reads `tile_x`, "
                "which nothing before it adds. Added by: slide_tiles. Hint: put "
                "a component that adds `tile_x` before it",
                "catalog/pipelines/a1.yaml:sinks[0]: `score_grid` reads `tile_y`, "
                "which nothing before it adds. Added by: slide_tiles. Hint: put "
                "a component that adds `tile_y` before it",
            ],
            id="frame fields the source lacks",
        ),
        pytest.param(
            {
                "pipelines/a1.yaml": """
                    source: {use: slide_tiles, tile_size: 64, mpp: 0.5}
                    steps: [{use: mean_intensity}, {use: mean_intensity}]
                    sinks: [{use: score_grid, output: heatmap}]
                """
            },
            [
                "catalog/pipelines/a1.yaml:steps[1]: per-run setting `scale` of "
                "`mean_intensity` is also a setting of `mean_intensity` at "
                "steps[0]. Settings are flat names, one per pipeline. Hint: drop "
                "one of the two components or use another in its place",
            ],
            id="per-run setting collision",
        ),
        pytest.param(
            {
                "pipelines/a1.yaml": HEATMAP.replace(
                    "[{use: score_grid, output: heatmap}]",
                    "[{use: score_grid, output: heatmap}, "
                    "{use: score_table, output: heatmap}]",
                )
            },
            [
                "catalog/pipelines/a1.yaml:sinks[1].output: output `heatmap` is "
                "also written by sinks[0]. One sink per output name. Hint: "
                "rename one `output:`",
            ],
            id="two sinks with one output",
        ),
    ],
)
def test_a_mistake_is_reported_where_it_is(
    root: Path, files: dict[str, str], expected: list[str]
) -> None:
    for name, text in files.items():
        write(root, name, text)

    assert errors(root) == expected


@pytest.mark.parametrize(
    "source", ["{use: slide_tiles, tile_size: true}", "{use: slide_tiles, mpp: .nan}"]
)
def test_a_source_option_of_the_wrong_kind_is_refused(root: Path, source: str) -> None:
    write(root, "pipelines/a1.yaml", HEATMAP.replace("{use: slide_tiles}", source))

    (error,) = errors(root)
    assert error.startswith("catalog/pipelines/a1.yaml:source.")
