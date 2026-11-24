"""The catalog loader: a resolved catalog or every error in the standard format.

Test-only components stand in for the real ones.
"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING

import pytest

from model_service.catalog import Catalog, CatalogInvalid, environments, load_catalog
from model_service.registry import Registry


if TYPE_CHECKING:
    from pathlib import Path

    from model_service.catalog.models import Model


REGISTRY = Registry.scan("tests.support.components")

MODEL = """
    loader: {use: local_file, path: w.onnx}
    runtime: {use: constant, value: 0.5, threads: 4}
    post: [{use: mean_intensity, channels: 2}]
    input: {tile_size: 8, mpp: 0.5}
    output: {kind: scores}
    batching: {max_size: 8, wait_s: 0}
    resources: {memory_gib: 1}
"""


@pytest.fixture
def root(tmp_path: Path) -> Path:
    catalog = tmp_path / "catalog"
    (catalog / "models").mkdir(parents=True)
    (catalog / "pipelines").mkdir()
    (tmp_path / "environments" / "test-env").mkdir(parents=True)
    return catalog


def write(root: Path, name: str, text: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))


def load(root: Path, environment: str | None = None) -> Catalog:
    folder = root.parent / "environments" / environment if environment else None
    return load_catalog(root, folder, registry=REGISTRY)


def errors(root: Path, environment: str | None = None) -> list[str]:
    with pytest.raises(CatalogInvalid) as raised:
        load(root, environment)
    return [str(error) for error in raised.value.errors]


def model(catalog: Catalog, model_id: str = "m1") -> Model:
    return catalog.models[model_id]


def test_an_empty_catalog_loads_for_the_base_and_every_environment(
    root: Path,
) -> None:
    assert [p.name for p in environments(root.parent / "environments")] == ["test-env"]
    for environment in [None, "test-env"]:
        catalog = load(root, environment)
        assert (catalog.models, catalog.pipelines) == ({}, {})


def test_a_document_resolves_its_components_with_their_options(root: Path) -> None:
    write(root, "models/m1.yaml", MODEL)

    resolved = model(load(root))

    assert resolved.input.tile_size == 8
    assert resolved.runtime.name == "constant"
    assert resolved.runtime.options.value == 0.5  # type: ignore[attr-defined]
    assert resolved.post[0].options.channels == 2  # type: ignore[attr-defined]


def test_every_mistake_in_every_file_is_reported_in_one_load(root: Path) -> None:
    write(
        root,
        "models/a.yaml",
        MODEL.replace("path: w.onnx", "").replace("value: 0.5", "valu: 0.5"),
    )
    write(
        root,
        "models/b.yaml",
        MODEL.replace("tile_size: 8", "tile_size: -1")
        .replace("value: 0.5", "value: 1.5")
        .replace("channels: 2", "channels: 5"),
    )

    assert errors(root) == [
        "catalog/models/a.yaml:loader.path: missing option `path`. "
        "path: A weights file on this machine. "
        "Hint: add `path: …` under `loader:`",
        "catalog/models/a.yaml:runtime.valu: unknown option `valu`. "
        "Options of `constant`: threads, value. Hint: did you mean `value`?",
        "catalog/models/b.yaml:input.tile_size: `-1` is invalid: input "
        "should be greater than 0. tile_size: Tile edge in pixels. "
        "Hint: change `tile_size`",
        "catalog/models/b.yaml:runtime.value: `1.5` is invalid: input "
        "should be less than or equal to 1. value: The score every tile gets. "
        "Hint: change `value`",
        "catalog/models/b.yaml:post[0].channels: `5` is invalid: input "
        "should be less than or equal to 3. channels: Colour channels to score. "
        "Hint: tiles are RGB, so pick 1, 2 or 3 channels",
    ]


def test_an_unknown_component_lists_what_fits_the_slot(root: Path) -> None:
    write(root, "models/m1.yaml", MODEL.replace("use: constant", "use: konstant"))

    assert errors(root) == [
        "catalog/models/m1.yaml:runtime.use: no component `konstant`. "
        "Usable under `runtime:`: constant. Hint: did you mean `constant`?",
    ]


def test_a_component_in_the_wrong_slot_is_refused_with_a_hint(root: Path) -> None:
    write(
        root,
        "models/m1.yaml",
        MODEL.replace(
            "{use: constant, value: 0.5, threads: 4}", "{use: mean_intensity}"
        ),
    )

    assert errors(root) == [
        "catalog/models/m1.yaml:runtime.use: `mean_intensity` is a step, "
        "not usable under `runtime:`. Usable under `runtime:`: constant. "
        "Hint: use `mean_intensity` under `pre:`, `post:` or `steps:`",
    ]


def test_a_missing_component_and_a_bare_name_are_errors(root: Path) -> None:
    write(
        root,
        "models/m1.yaml",
        MODEL.replace("loader: {use: local_file, path: w.onnx}", "").replace(
            "{use: constant, value: 0.5, threads: 4}", "constant"
        ),
    )

    assert errors(root)[1:] == [
        "catalog/models/m1.yaml:loader: missing `loader:`. "
        "A loader, e.g. `loader: {use: local_file, …}`. "
        "Hint: add `loader:` with `use:` and its options",
        "catalog/models/m1.yaml:runtime: `constant` is not a mapping. "
        "`runtime: {use: <name>, <option>: <value>, …}`. "
        "Hint: write `runtime: {use: constant}`",
    ]


def test_an_environment_patch_overrides_only_what_it_names(root: Path) -> None:
    write(root, "models/m1.yaml", MODEL)
    write(
        root.parent,
        "environments/test-env/models/m1.yaml",
        "runtime: {value: 0.9}\n",
    )

    options = model(load(root, "test-env")).runtime.options

    assert (options.value, options.threads) == (0.9, 4)  # type: ignore[attr-defined]
    assert model(load(root)).runtime.options.value == 0.5  # type: ignore[attr-defined]


def test_a_null_in_a_patch_removes_the_key(root: Path) -> None:
    write(root, "models/m1.yaml", MODEL)
    write(root.parent, "environments/test-env/models/m1.yaml", "post: null\n")

    assert model(load(root, "test-env")).post == ()


def test_a_mistake_in_a_patched_value_names_the_patch_file(root: Path) -> None:
    write(root, "models/m1.yaml", MODEL)
    write(
        root.parent,
        "environments/test-env/models/m1.yaml",
        "runtime: {value: 7}\n",
    )

    assert errors(root, "test-env") == [
        "environments/test-env/models/m1.yaml:runtime.value: `7` is "
        "invalid: input should be less than or equal to 1. "
        "value: The score every tile gets. Hint: change `value`",
    ]


def test_a_patch_without_a_base_file_is_an_error(root: Path) -> None:
    write(root, "models/m1.yaml", MODEL)
    write(
        root.parent,
        "environments/test-env/models/m2.yaml",
        "runtime: {value: 0.9}\n",
    )

    assert errors(root, "test-env") == [
        "environments/test-env/models/m2.yaml: patches a file that "
        "does not exist. Base files in models/: m1.yaml. "
        "Hint: name the patch after the file it changes",
    ]


def test_a_patch_outside_the_patch_folders_is_an_error(root: Path) -> None:
    write(root.parent, "environments/test-env/model/m1.yaml", "post: null\n")
    write(root.parent, "environments/test-env/values.yaml", "workers: []\n")
    write(root.parent, "environments/test-env/workers/cpu.yaml", "groupName: cpu\n")

    assert errors(root, "test-env") == [
        "environments/test-env/model/m1.yaml: not a patch folder. Patch "
        "folders: models/, pipelines/, beside values.yaml, workers/, k8s/. "
        "Hint: move the file into one of them",
    ]


def test_an_unknown_environment_is_an_error(root: Path) -> None:
    assert errors(root, "prod") == [
        "environments/prod: no such environment. Environments: test-env. "
        "Hint: create environments/prod/ or fix the name",
    ]


@pytest.mark.parametrize(
    ("name", "text", "expected"),
    [
        (
            "models/m1.yaml",
            "input: {tile_size: 8}\nruntime: use: constant\n",
            "catalog/models/m1.yaml: not valid YAML at line 2, column 13: "
            "mapping values are not allowed here. A mapping of keys to values. "
            "Hint: fix the syntax near line 2",
        ),
        (
            "models/m1.yaml",
            "- 8\n",
            "catalog/models/m1.yaml: the file is a list, not a mapping. A "
            "mapping of keys to values. Hint: write `key: value` lines at the top "
            "level",
        ),
        (
            "models/My Model.yaml",
            MODEL,
            "catalog/models/My Model.yaml: `My Model` is not a valid ID. "
            "Lowercase letters, digits and single hyphens, e.g. `my-model`. "
            "Hint: rename the file; its name is the public ID",
        ),
        (
            "modelz/m1.yaml",
            MODEL,
            "catalog/modelz/m1.yaml: not a catalog folder. Folders: models/, "
            "pipelines/. Hint: move the file into one of them",
        ),
    ],
)
def test_files_that_cannot_be_read_as_documents_are_errors(
    root: Path, name: str, text: str, expected: str
) -> None:
    write(root, name, text)

    assert errors(root) == [expected]


def test_an_pipeline_calling_a_broken_model_gets_no_second_error(root: Path) -> None:
    write(root, "models/m1.yaml", MODEL)
    write(root, "models/broken.yaml", "input: {tile_size: 8}\n")
    pipeline = """
        source: {{use: grid}}
        steps: [{{model: {model}}}]
        sinks: [{{use: score_table, output: scores}}]
    """
    write(root, "pipelines/on-broken.yaml", pipeline.format(model="broken"))
    write(root, "pipelines/unknown.yaml", pipeline.format(model="m9"))

    assert [e for e in errors(root) if "pipelines/" in e] == [
        "catalog/pipelines/unknown.yaml:steps[0].model: no model `m9`. "
        "Models: m1. Hint: add catalog/models/m9.yaml or fix the name",
    ]


def test_an_id_ending_in_a_newline_is_refused(root: Path) -> None:
    write(root, "models/m1\n.yaml", MODEL)

    (error,) = errors(root)
    assert "is not a valid ID" in error


def test_a_file_that_is_not_utf8_is_an_error(root: Path) -> None:
    (root / "models" / "m1.yaml").write_bytes(b"input: 8\nname: caf\xe9\n")

    assert errors(root) == [
        "catalog/models/m1.yaml: not UTF-8 text: byte 0xe9 on line 2. A "
        "mapping of keys to values. Hint: save the file as UTF-8"
    ]


def test_a_folder_named_like_a_document_is_an_error(root: Path) -> None:
    (root / "models" / "m1.yaml").mkdir()

    assert errors(root) == [
        "catalog/models/m1.yaml: a folder, not a file. A mapping of keys "
        "to values. Hint: rename or remove the folder"
    ]
