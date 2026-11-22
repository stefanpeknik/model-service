"""Analysis files: what they resolve to and every load-time check."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from model_service.builders.implementations.heatmap.spec import HeatmapConfiguration
from model_service.catalog import CatalogInvalid, load_catalog
from model_service.components import Col, Registry, default_registry


if TYPE_CHECKING:
    from model_service.catalog.analyses import Analysis


CATALOG = Path(__file__).resolve().parents[2] / "catalog"
REGISTRY = Registry(
    {
        (c.kind, c.name): c
        for c in [*default_registry(), *Registry.scan("tests.support.components")]
    }
)


def model(tile_size: int = 64, mpp: float = 0.5, output: str = "kind: scores") -> str:
    return f"""
        weights: {{use: local, path: /weights/model.onnx}}
        runtime: {{use: onnx}}
        input: {{tile_size: {tile_size}, mpp: {mpp}}}
        output: {{{output}}}
        batching: {{max_size: 8, wait_s: 0.01}}
        resources: {{memory_gib: 2}}
    """


HEATMAP = """
    source: {use: slide_tiles}
    steps: [{model: m1}]
    sinks: [{use: score_grid, output: heatmap}]
"""


@pytest.fixture
def root(tmp_path: Path) -> Path:
    catalog = tmp_path / "catalog"
    write(catalog, "served-models/m1.yaml", model())
    return catalog


def write(root: Path, name: str, text: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))


def analysis(root: Path, analysis_id: str = "a1") -> Analysis:
    return load_catalog(root, registry=REGISTRY).entries("analyses")[analysis_id].value


def errors(root: Path) -> list[str]:
    with pytest.raises(CatalogInvalid) as raised:
        load_catalog(root, registry=REGISTRY)
    return [str(error) for error in raised.value.errors]


@pytest.mark.parametrize("environment", [None, "rationai-test"])
def test_prostate_heatmap_is_todays_heatmap_builder_on_prostate(
    environment: str | None,
) -> None:
    catalog = load_catalog(CATALOG, environment)
    prostate = catalog.entries("served-models")["prostate-classifier-1"].value
    heatmap = catalog.entries("analyses")["prostate-heatmap"].value

    assert heatmap.settings() == HeatmapConfiguration().model_dump()
    assert heatmap.inputs == {"slide": True, "region_of_interest": False}
    assert list(heatmap.models) == ["prostate-classifier-1"]
    assert (heatmap.source.options.tile_size, heatmap.source.options.mpp) == (
        prostate.input.tile_size,
        prostate.input.mpp,
    )
    assert heatmap.columns["scores"] == Col("float32", (1, 1, 1))
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
        "analyses/a1.yaml",
        HEATMAP.replace(
            "{use: slide_tiles}", f"{{use: slide_tiles, region_of_interest: {roi}}}"
        ),
    )

    assert analysis(root).inputs == inputs


def test_a_caller_overrides_per_run_settings_and_nothing_else(root: Path) -> None:
    write(root, "analyses/a1.yaml", HEATMAP)
    resolved = analysis(root)

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
            {"analyses/a1.yaml": HEATMAP.replace("model: m1", "model: m2")},
            [
                "catalog/analyses/a1.yaml:steps[0].model: no served model `m2`. "
                "Served models: m1. Hint: add catalog/served-models/m2.yaml or "
                "fix the name"
            ],
            id="unknown model",
        ),
        pytest.param(
            {
                "analyses/a1.yaml": HEATMAP.replace(
                    "{use: slide_tiles}", "{use: slide_tiles, tile_size: 128}"
                )
            },
            [
                "catalog/analyses/a1.yaml:source.tile_size: `128`, but model step "
                "`m1` takes 64. The first model step's value. Hint: leave "
                "`tile_size` and `mpp` out so the first model step sets them; a "
                "model-free analysis sets both"
            ],
            id="geometry conflicts with the model",
        ),
        pytest.param(
            {
                "served-models/m2.yaml": model(mpp=1.0),
                "analyses/a1.yaml": HEATMAP.replace(
                    "[{model: m1}]", "[{model: m1}, {model: m2}]"
                ),
            },
            [
                "catalog/analyses/a1.yaml:steps[1].model: model step `m2` takes 64 "
                "px tiles at 1.0 µm/px, but the first model step `m1` takes 64 px "
                "at 0.5 µm/px. Model steps that read the same tiles. Hint: an "
                "analysis reads one tile geometry; split it into two analyses"
            ],
            id="two model steps with different geometry",
        ),
        pytest.param(
            {
                "analyses/a1.yaml": """
                    source: {use: slide_tiles}
                    sinks: [{use: score_grid, output: heatmap}]
                """
            },
            [
                "catalog/analyses/a1.yaml:source: a model-free analysis must set "
                "`tile_size` and `mpp`. E.g. `source: {use: slide_tiles, "
                "tile_size: 256, mpp: 1.0}`. Hint: leave `tile_size` and `mpp` "
                "out so the first model step sets them; a model-free analysis "
                "sets both",
                "catalog/analyses/a1.yaml:sinks[0]: `score_grid` reads `scores`, "
                "which nothing before it adds. Added by: a model step with "
                "`output: {kind: scores}`, mean_intensity. Hint: put a component "
                "that adds `scores` before it",
            ],
            id="model-free without geometry or scores",
        ),
        pytest.param(
            {
                "served-models/m1.yaml": model(output="kind: scores, size: 4"),
                "analyses/a1.yaml": HEATMAP.replace(
                    "use: score_grid, output: heatmap", "use: score_table, output: t"
                ),
            },
            [
                "catalog/analyses/a1.yaml:sinks[0]: `score_table` reads `scores` "
                "as float32[?,1,1], but model step `m1` adds float32[1,4,4] "
                "(model m1). A column float32[?,1,1]. Hint: pick a model or "
                "step whose `scores` fits, or a sink that reads float32[1,4,4]",
            ],
            id="column shape mismatch",
        ),
        pytest.param(
            {
                "analyses/a1.yaml": """
                    source: {use: grid, tile_size: 64}
                    steps: [{model: m1}]
                    sinks: [{use: score_grid, output: heatmap}]
                """
            },
            [
                "catalog/analyses/a1.yaml:sinks[0]: `score_grid` reads frame "
                "fields level, mpp_x, mpp_y, stride_x, stride_y, which `grid` "
                "doesn't give. Frame fields of `grid`: extent_x, extent_y, "
                "tile_extent_x, tile_extent_y. Hint: use a source that gives "
                "them, e.g. slide_tiles",
                "catalog/analyses/a1.yaml:sinks[0]: `score_grid` reads `tile_x`, "
                "which nothing before it adds. Added by: slide_tiles. Hint: put "
                "a component that adds `tile_x` before it",
                "catalog/analyses/a1.yaml:sinks[0]: `score_grid` reads `tile_y`, "
                "which nothing before it adds. Added by: slide_tiles. Hint: put "
                "a component that adds `tile_y` before it",
            ],
            id="frame fields the source lacks",
        ),
        pytest.param(
            {
                "analyses/a1.yaml": """
                    source: {use: slide_tiles, tile_size: 64, mpp: 0.5}
                    steps: [{use: mean_intensity}, {use: mean_intensity}]
                    sinks: [{use: score_grid, output: heatmap}]
                """
            },
            [
                "catalog/analyses/a1.yaml:steps[1]: per-run setting `scale` of "
                "`mean_intensity` is also a setting of `mean_intensity` at "
                "steps[0]. Settings are flat names, one per analysis. Hint: drop "
                "one of the two components or use another in its place",
            ],
            id="per-run setting collision",
        ),
        pytest.param(
            {
                "analyses/a1.yaml": HEATMAP.replace(
                    "[{use: score_grid, output: heatmap}]",
                    "[{use: score_grid, output: heatmap}, "
                    "{use: score_table, output: heatmap}]",
                )
            },
            [
                "catalog/analyses/a1.yaml:sinks[1].output: output `heatmap` is "
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
