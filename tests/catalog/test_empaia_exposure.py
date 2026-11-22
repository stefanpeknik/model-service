"""`expose.empaia`: an analysis becomes one EMPAIA app, checked when it loads."""

from __future__ import annotations

from pathlib import Path

import pytest

from model_service.catalog import CatalogInvalid, load_catalog
from model_service.providers.empaia.ead.vocabulary import AppMode, Trigger
from tests.support.analysis_catalog import REGISTRY, write
from tests.support.analysis_catalog import served_model as model


CATALOG = Path(__file__).resolve().parents[2] / "catalog"

APP = """
    source: {use: slide_tiles}
    steps: [{model: m1}]
    sinks: [{use: score_grid, output: heatmap}]
    expose:
      empaia:
        name: Test heatmap
        name_short: TestHM
        description: Finds things.
        modes:
          standalone:
          preprocessing:
            triggers: [{tissue: PROSTATE, stain: H_AND_E}]
        outputs:
          heatmap: {type: continuous_pixelmap}
"""


@pytest.fixture
def root(tmp_path: Path) -> Path:
    catalog = tmp_path / "catalog"
    write(catalog, "served-models/m1.yaml", model())
    return catalog


def errors(root: Path) -> list[str]:
    with pytest.raises(CatalogInvalid) as raised:
        load_catalog(root, registry=REGISTRY)
    return [str(error) for error in raised.value.errors]


@pytest.mark.parametrize("environment", [None, "rationai-test"])
def test_the_prostate_heatmap_is_todays_prostate_app(environment: str | None) -> None:
    analysis = load_catalog(CATALOG, environment).entries("analyses")
    app = analysis["prostate-heatmap"].value.empaia

    assert (app.name, app.name_short) == ("Prostate cancer heatmap", "ProstHM")
    assert app.modes == (AppMode.STANDALONE, AppMode.PREPROCESSING)
    assert app.triggers == (Trigger(tissue="PROSTATE", stain="H_AND_E"),)
    assert app.parameters == ()
    assert {k: c.name for k, c in app.inputs.items()} == {
        "slide": "wsi",
        "region_of_interest": "rectangle",
    }
    assert app.slide == "slide"
    assert {k: p.name for k, p in app.outputs.items()} == {
        "heatmap": "continuous_pixelmap"
    }
    assert app.viewer.outputs["heatmap"].shader.type == "heatmap"  # type: ignore[union-attr]


def test_only_the_prostate_heatmap_is_offered_in_empaia() -> None:
    analyses = load_catalog(CATALOG).entries("analyses")

    assert [i for i, e in analyses.items() if e.value.empaia] == ["prostate-heatmap"]


def test_standalone_parameters_are_per_run_settings(root: Path) -> None:
    write(
        root,
        "analyses/a1.yaml",
        APP.replace("standalone:", "standalone: {parameters: [stride_fraction]}"),
    )

    app = load_catalog(root, registry=REGISTRY).entries("analyses")["a1"].value.empaia

    assert app.parameters == ("stride_fraction",)


def test_an_analysis_without_expose_is_no_app(root: Path) -> None:
    write(root, "analyses/a1.yaml", APP[: APP.index("    expose:")])

    analysis = load_catalog(root, registry=REGISTRY).entries("analyses")["a1"].value

    assert analysis.empaia is None


PREFIX = "catalog/analyses/a1.yaml:expose.empaia"


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        pytest.param(
            ("{use: slide_tiles}", "{use: slide_tiles, region_of_interest: required}"),
            [
                f"{PREFIX}.modes.preprocessing: preprocessing starts a job with "
                "the slide alone, but a1 requires region_of_interest. Standalone "
                "mode only, or an optional input. Hint: drop `preprocessing:` or "
                "make the input optional in the source"
            ],
            id="preprocessing with a required region of interest",
        ),
        pytest.param(
            ("Test heatmap", "A name far longer than EMPAIA takes"),
            [
                f"{PREFIX}.name: `A name far longer than EMPAIA takes` is "
                "invalid: string should have at most 30 characters. name: The "
                "app's name in EMPAIA (3 to 30 letters, digits, spaces, `-`, "
                "`_`). Hint: change `name`"
            ],
            id="name over EMPAIA's limit",
        ),
        pytest.param(
            ("{type: continuous_pixelmap}", "{type: nominal_pixelmap}"),
            [
                f"{PREFIX}.outputs.heatmap.type: no publisher takes `score_grid` "
                "as EMPAIA type `nominal_pixelmap`. Types for `score_grid`: "
                "continuous_pixelmap. Hint: pick one of them, or leave the output "
                "out of EMPAIA"
            ],
            id="an output type nothing publishes",
        ),
        pytest.param(
            ("heatmap: {type:", "scores: {type:"),
            [
                f"{PREFIX}.outputs.scores: a1 has no output `scores`. Outputs: "
                "heatmap. Hint: list sink outputs only"
            ],
            id="an output no sink writes",
        ),
        pytest.param(
            ("standalone:", "standalone: {parameters: [tile_size]}"),
            [
                f"{PREFIX}.modes.standalone.parameters[0]: `tile_size` is not a "
                "setting of a1. Settings: min_tissue_fraction, skip_background, "
                "stride_fraction. Hint: list per-run settings (`PerRun` options) "
                "only"
            ],
            id="a parameter that is no setting",
        ),
        pytest.param(
            (
                "outputs:",
                "inputs:\n          slide: {type: rectangle}\n        outputs:",
            ),
            [
                f"{PREFIX}.inputs: an EMPAIA app needs exactly one `wsi` input, "
                "found 0. One input of type `wsi`, the slide. Hint: declare the "
                "slide input as `{type: wsi}` and no other"
            ],
            id="no slide",
        ),
    ],
)
def test_a_mistake_is_reported_where_it_is(
    root: Path, change: tuple[str, str], expected: list[str]
) -> None:
    write(root, "analyses/a1.yaml", APP.replace(*change))

    assert errors(root) == expected


def test_a_multi_channel_grid_is_no_continuous_pixelmap(root: Path) -> None:
    write(root, "served-models/m1.yaml", model(output="kind: scores, channels: 2"))
    write(root, "analyses/a1.yaml", APP)

    assert errors(root) == [
        f"{PREFIX}.outputs.heatmap.type: a continuous pixelmap carries one "
        "channel; the scores have 2. A model step whose scores have one channel. "
        "Hint: change the model or the output's type"
    ]
