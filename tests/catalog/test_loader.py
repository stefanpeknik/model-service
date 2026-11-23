"""The catalog loader: a resolved catalog or every error in the standard format.

Test-only components and document kinds stand in for the real ones.
"""

from __future__ import annotations

import textwrap
from typing import TYPE_CHECKING, Any

import pytest

from model_service.catalog import Catalog, CatalogInvalid, environments, load_catalog
from model_service.registry import Registry
from tests.support.catalog_kinds import TEST_KINDS


if TYPE_CHECKING:
    from pathlib import Path


REGISTRY = Registry.scan("tests.support.components")

MODEL = """
    tile_size: 8
    weights: {use: local_file, path: w.onnx}
    runtime: {use: constant, value: 0.5, threads: 4}
    post: [{use: mean_intensity, channels: 2}]
"""


@pytest.fixture
def root(tmp_path: Path) -> Path:
    catalog = tmp_path / "catalog"
    (catalog / "models").mkdir(parents=True)
    (catalog / "chains").mkdir()
    (catalog / "environments" / "test-env").mkdir(parents=True)
    return catalog


def write(root: Path, name: str, text: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))


def load(root: Path, environment: str | None = None) -> Catalog:
    return load_catalog(root, environment, kinds=TEST_KINDS, registry=REGISTRY)


def errors(root: Path, environment: str | None = None) -> list[str]:
    with pytest.raises(CatalogInvalid) as raised:
        load(root, environment)
    return [str(error) for error in raised.value.errors]


def model(catalog: Catalog, model_id: str = "m1") -> Any:
    return catalog.entries("models")[model_id].value


def test_an_empty_catalog_loads_for_the_base_and_every_environment(
    root: Path,
) -> None:
    assert environments(root) == ["test-env"]
    for environment in [None, "test-env"]:
        assert load(root, environment).entries("models") == {}


def test_a_document_resolves_its_components_with_their_options(root: Path) -> None:
    write(root, "models/m1.yaml", MODEL)

    resolved = model(load(root))

    assert resolved.tile_size == 8
    assert resolved.runtime.name == "constant"
    assert resolved.runtime.options.value == 0.5
    assert resolved.post[0].options.channels == 2


def test_every_mistake_in_every_file_is_reported_in_one_load(root: Path) -> None:
    write(
        root,
        "models/a.yaml",
        """
        tile_size: 8
        weights: {use: local_file}
        runtime: {use: constant, valu: 0.5}
        """,
    )
    write(
        root,
        "models/b.yaml",
        """
        tile_size: -1
        weights: {use: local_file, path: w.onnx}
        runtime: {use: constant, value: 1.5}
        post: [{use: mean_intensity, channels: 5}]
        """,
    )

    assert errors(root) == [
        "catalog/models/a.yaml:weights.path: missing option `path`. "
        "path: A weights file on this machine. "
        "Hint: add `path: …` under `weights:`",
        "catalog/models/a.yaml:runtime.valu: unknown option `valu`. "
        "Options of `constant`: threads, value. Hint: did you mean `value`?",
        "catalog/models/b.yaml:tile_size: `-1` is invalid: input should be greater "
        "than 0. tile_size: Tile edge the model takes. Hint: change `tile_size`",
        "catalog/models/b.yaml:runtime.value: `1.5` is invalid: input should be "
        "less than or equal to 1. value: The score every tile gets. "
        "Hint: change `value`",
        "catalog/models/b.yaml:post[0].channels: `5` is invalid: input should be "
        "less than or equal to 3. channels: Colour channels to score. "
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
        "catalog/models/m1.yaml:runtime.use: `mean_intensity` is a step, not usable "
        "under `runtime:`. Usable under `runtime:`: constant. "
        "Hint: use `mean_intensity` under `pre:`, `post:` or `steps:`",
    ]


def test_a_missing_component_and_a_bare_name_are_errors(root: Path) -> None:
    write(root, "models/m1.yaml", "tile_size: 8\nruntime: constant\n")

    assert errors(root) == [
        "catalog/models/m1.yaml:weights: missing `weights:`. "
        "A loader, e.g. `weights: {use: local_file, …}`. "
        "Hint: add `weights:` with `use:` and its options",
        "catalog/models/m1.yaml:runtime: `constant` is not a mapping. "
        "`runtime: {use: <name>, <option>: <value>, …}`. "
        "Hint: write `runtime: {use: constant}`",
    ]


def test_an_environment_patch_overrides_only_what_it_names(root: Path) -> None:
    write(root, "models/m1.yaml", MODEL)
    write(root, "environments/test-env/models/m1.yaml", "runtime: {value: 0.9}\n")

    patched = model(load(root, "test-env"))

    assert (patched.runtime.options.value, patched.runtime.options.threads) == (0.9, 4)
    assert model(load(root)).runtime.options.value == 0.5


def test_a_null_in_a_patch_removes_the_key(root: Path) -> None:
    write(root, "models/m1.yaml", MODEL)
    write(root, "environments/test-env/models/m1.yaml", "post: null\n")

    assert model(load(root, "test-env")).post == []


def test_a_mistake_in_a_patched_value_names_the_patch_file(root: Path) -> None:
    write(root, "models/m1.yaml", MODEL)
    write(root, "environments/test-env/models/m1.yaml", "runtime: {value: 7}\n")

    assert errors(root, "test-env") == [
        "catalog/environments/test-env/models/m1.yaml:runtime.value: `7` is "
        "invalid: input should be less than or equal to 1. "
        "value: The score every tile gets. Hint: change `value`",
    ]


def test_a_patch_without_a_base_file_is_an_error(root: Path) -> None:
    write(root, "models/m1.yaml", MODEL)
    write(root, "environments/test-env/models/m2.yaml", "runtime: {value: 0.9}\n")

    assert errors(root, "test-env") == [
        "catalog/environments/test-env/models/m2.yaml: patches a file that does "
        "not exist. Base files in models/: m1.yaml. "
        "Hint: name the patch after the file it changes",
    ]


def test_an_unknown_environment_is_an_error(root: Path) -> None:
    assert errors(root, "prod") == [
        "catalog/environments/prod: no such environment. Environments: test-env. "
        "Hint: create catalog/environments/prod/ or fix the name",
    ]


@pytest.mark.parametrize(
    ("name", "text", "expected"),
    [
        (
            "models/m1.yaml",
            "tile_size: 8\nruntime: use: constant\n",
            "catalog/models/m1.yaml: not valid YAML at line 2, column 13: mapping "
            "values are not allowed here. A mapping of keys to values. "
            "Hint: fix the syntax near line 2",
        ),
        (
            "models/m1.yaml",
            "- 8\n",
            "catalog/models/m1.yaml: the file is a list, not a mapping. A mapping "
            "of keys to values. Hint: write `key: value` lines at the top level",
        ),
        (
            "models/My Model.yaml",
            MODEL,
            "catalog/models/My Model.yaml: `My Model` is not a valid ID. Lowercase "
            "letters, digits and single hyphens, e.g. `my-model`. "
            "Hint: rename the file; its name is the public ID",
        ),
        (
            "model/m1.yaml",
            MODEL,
            "catalog/model/m1.yaml: not a catalog folder. Folders: chains/, "
            "models/, environments/<env>/. Hint: move the file into one of them",
        ),
    ],
)
def test_files_that_cannot_be_read_as_documents_are_errors(
    root: Path, name: str, text: str, expected: str
) -> None:
    write(root, name, text)

    assert errors(root) == [expected]


def test_a_later_kind_sees_the_documents_resolved_before_it(root: Path) -> None:
    write(root, "models/m1.yaml", MODEL)
    write(root, "models/broken.yaml", "tile_size: 8\n")
    chain = """
        model: {model}
        source: {{use: grid}}
        sinks: [{{use: score_table, output: scores}}]
    """
    write(root, "chains/good.yaml", chain.format(model="m1"))
    write(root, "chains/on-broken.yaml", chain.format(model="broken"))
    write(root, "chains/unknown.yaml", chain.format(model="m9"))

    assert [e for e in errors(root) if "chains/" in e] == [
        "catalog/chains/unknown.yaml:model: no test model `m9`. "
        "Test models: m1. Hint: add catalog/models/m9.yaml or fix the name",
    ]


def test_an_id_ending_in_a_newline_is_refused(root: Path) -> None:
    write(root, "models/m1\n.yaml", MODEL)

    (error,) = errors(root)
    assert "is not a valid ID" in error


def test_a_file_that_is_not_utf8_is_an_error(root: Path) -> None:
    (root / "models" / "m1.yaml").write_bytes(b"tile_size: 8\nname: caf\xe9\n")

    assert errors(root) == [
        "catalog/models/m1.yaml: not UTF-8 text: byte 0xe9 on line 2. A mapping "
        "of keys to values. Hint: save the file as UTF-8"
    ]


def test_a_folder_named_like_a_document_is_an_error(root: Path) -> None:
    (root / "models" / "m1.yaml").mkdir()

    assert errors(root) == [
        "catalog/models/m1.yaml: a folder, not a file. A mapping of keys to "
        "values. Hint: rename or remove the folder"
    ]
