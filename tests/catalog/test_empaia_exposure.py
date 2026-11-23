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
def test_the_prostate_heatmap_is_the_prostate_app(environment: str | None) -> None:
    analysis = load_catalog(CATALOG, environment).analyses
    app = analysis["prostate-heatmap"].empaia

    # EMPAIA knows the app by these; changing them registers a new app.
    assert (app.name, app.name_short) == ("Prostate cancer heatmap", "ProstHM")
    assert app.modes == (AppMode.STANDALONE, AppMode.PREPROCESSING)
    assert app.triggers == (Trigger(tissue="PROSTATE", stain="H_AND_E"),)
    assert app.slide == "slide"


def test_standalone_parameters_are_per_run_settings(root: Path) -> None:
    write(
        root,
        "analyses/a1.yaml",
        APP.replace("standalone:", "standalone: {parameters: [stride_fraction]}"),
    )

    app = load_catalog(root, registry=REGISTRY).analyses["a1"].empaia

    assert app.parameters == ("stride_fraction",)


def test_an_analysis_without_expose_is_no_app(root: Path) -> None:
    write(root, "analyses/a1.yaml", APP[: APP.index("    expose:")])

    analysis = load_catalog(root, registry=REGISTRY).analyses["a1"]

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
                "continuous_pixelmap, string. Hint: pick one of them, or leave the output "
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
        pytest.param(
            (
                "outputs:\n",
                "viewer: {outputs: {heatmp: {render: shader}}}\n        outputs:\n",
            ),
            [
                f"{PREFIX}.viewer.outputs.heatmp: the viewer names output "
                "`heatmp`, which the app does not publish. Published outputs: "
                "heatmap. Hint: use an output listed under `outputs:`"
            ],
            id="a viewer entry for no output",
        ),
        pytest.param(
            (
                "outputs:\n",
                "viewer: {inputs: {slid: {lossless: true}}}\n        outputs:\n",
            ),
            [
                f"{PREFIX}.viewer.inputs.slid: the viewer names input `slid`, "
                "which the app does not take. App inputs: region_of_interest, "
                "slide. Hint: use an input or a standalone parameter"
            ],
            id="a viewer entry for no input",
        ),
        pytest.param(
            (
                "outputs:\n",
                "viewer: {outputs: {heatmap: {shader: {type: heatmap, params: "
                "{color: {r: 1}}}}}}\n        outputs:\n",
            ),
            [f"{PREFIX}.viewer.outputs.heatmap.shader.params: "],
            id="a nested shader parameter",
        ),
    ],
)
def test_a_mistake_is_reported_where_it_is(
    root: Path, change: tuple[str, str], expected: list[str]
) -> None:
    write(root, "analyses/a1.yaml", APP.replace(*change))

    found = errors(root)

    assert [e[: len(x)] for e, x in zip(found, expected, strict=True)] == expected


def test_a_multi_channel_grid_is_no_continuous_pixelmap(root: Path) -> None:
    write(root, "served-models/m1.yaml", model(output="kind: scores, channels: 2"))
    write(root, "analyses/a1.yaml", APP)

    assert errors(root) == [
        f"{PREFIX}.outputs.heatmap.type: a continuous pixelmap carries one "
        "channel; the scores have 2. A model step whose scores have one channel. "
        "Hint: change the model or the output's type"
    ]


def test_an_input_type_no_converter_takes_is_named(root: Path) -> None:
    write(
        root,
        "analyses/a1.yaml",
        APP.replace(
            "outputs:",
            "inputs: {region_of_interest: {type: polygon}}\n        outputs:",
        ),
    )

    assert errors(root) == [
        f"{PREFIX}.inputs.region_of_interest.type: no converter takes EMPAIA "
        "type `polygon`. Input types: primitive, rectangle, string, wsi. Hint: pick a type "
        "one converter takes"
    ]


def test_standalone_parameters_need_a_primitive_converter(root: Path) -> None:
    write(
        root,
        "analyses/a1.yaml",
        APP.replace("standalone:", "standalone: {parameters: [stride_fraction]}"),
    )

    with pytest.raises(CatalogInvalid) as raised:
        load_catalog(root)

    assert [str(e) for e in raised.value.errors] == [
        f"{PREFIX}.modes.standalone.parameters: no converter takes EMPAIA type "
        "`primitive`, which standalone parameters are offered as. Input types: "
        "rectangle, wsi. Hint: leave out `parameters`, or add a converter of "
        "type `primitive`"
    ]
