"""Served-model files: what they resolve to, their digest and their mistakes."""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from model_service.catalog import CatalogInvalid, load_catalog


if TYPE_CHECKING:
    from model_service.catalog.served_models import ServedModel


REPO = Path(__file__).resolve().parents[2]
CATALOG = REPO / "catalog"

MODEL = """
    loader: {use: local, path: /weights/model.onnx}
    runtime: {use: onnx}
    input: {tile_size: 64, mpp: 0.5}
    output: {kind: scores, channels: 1, size: 1, range: [0, 1]}
    batching: {max_size: 8, wait_s: 0.01}
    resources: {cpus: 1, memory_gib: 2}
"""


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "catalog"


def write(root: Path, name: str, text: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text))


def served(root: Path, model_id: str = "m1") -> ServedModel:
    return load_catalog(root).served_models[model_id]


def test_the_digest_follows_what_the_model_computes_not_how_it_is_scheduled(
    root: Path,
) -> None:
    write(root, "served-models/m1.yaml", MODEL)
    digest = served(root).digest

    def digest_with(text: str) -> str:
        write(root, "served-models/m1.yaml", text)
        return served(root).digest

    assert digest_with(MODEL.replace("{use: onnx}", "{use: onnx, threads: 1}")) == (
        digest
    )
    assert digest_with(MODEL.replace("cpus: 1", "cpus: 2, gpus: 1")) == digest
    assert digest_with(MODEL.replace("wait_s: 0.01", "wait_s: 0.5")) == digest
    assert digest_with(MODEL + "    serving: {max_replicas: 1}\n") == digest
    assert digest_with(MODEL.replace("model.onnx", "model-2.onnx")) != digest
    assert digest_with(MODEL.replace("max_size: 8", "max_size: 16")) != digest
    assert digest_with(MODEL.replace("{use: onnx}", "{use: onnx, fp16: false}")) != (
        digest
    )


def test_a_wrong_value_and_an_unknown_key_are_refused_with_what_fits(
    root: Path,
) -> None:
    write(
        root,
        "served-models/m1.yaml",
        MODEL.replace("kind: scores", "kind: score") + "    contract: spatial-scores\n",
    )

    with pytest.raises(CatalogInvalid) as raised:
        load_catalog(root)

    assert [str(e) for e in raised.value.errors] == [
        "catalog/served-models/m1.yaml:output.kind: `score` is invalid: input "
        "should be 'scores' or 'embedding'. output: `{kind: scores, …}` or "
        "`{kind: embedding, summaries: […]}`. Hint: change `kind`",
        "catalog/served-models/m1.yaml:contract: unknown key `contract`. Keys of "
        "this file: batching, input, loader, output, post, pre, resources, runtime, "
        "serving. Hint: remove it",
    ]


def test_the_onnx_runtime_always_applies_its_session_options(root: Path) -> None:
    runtime = "{use: onnx, apply_session_options: false}"
    write(root, "served-models/m1.yaml", MODEL.replace("{use: onnx}", runtime))

    with pytest.raises(CatalogInvalid) as raised:
        load_catalog(root)

    (error,) = raised.value.errors
    assert "unknown option `apply_session_options`" in str(error)


def test_a_hugging_face_revision_must_be_a_commit(root: Path) -> None:
    write(
        root,
        "served-models/m1.yaml",
        MODEL.replace(
            "{use: local, path: /weights/model.onnx}",
            "{use: huggingface, repo: paige-ai/Virchow2, revision: main}",
        ),
    )

    with pytest.raises(CatalogInvalid) as raised:
        load_catalog(root)

    assert [str(e) for e in raised.value.errors] == [
        "catalog/served-models/m1.yaml:loader.revision: `main` is invalid: string "
        "should match pattern '^[0-9a-f]{40}$'. revision: The full 40-character "
        "commit hash to download. A branch or tag can move, so the weights would "
        "change under the same file. Hint: copy the full commit hash from the "
        "history of the repository's files on huggingface.co"
    ]


@pytest.mark.parametrize(
    ("old", "new", "where"),
    [
        ("range: [0, 1]", "range: [0, .nan]", "output.range[1]"),
        ("range: [0, 1]", "range: [-.inf, 1]", "output.range[0]"),
        ("wait_s: 0.01", "wait_s: .inf", "batching.wait_s"),
        ("memory_gib: 2", "memory_gib: .nan", "resources.memory_gib"),
        ("cpus: 1", "cpus: .inf", "resources.cpus"),
        ("tile_size: 64", "tile_size: true", "input.tile_size"),
        ("max_size: 8", "max_size: 8.0", "batching.max_size"),
    ],
)
def test_values_of_the_wrong_kind_are_refused(
    root: Path, old: str, new: str, where: str
) -> None:
    write(root, "served-models/m1.yaml", MODEL.replace(old, new))

    with pytest.raises(CatalogInvalid) as raised:
        load_catalog(root)

    (error,) = raised.value.errors
    assert f"m1.yaml:{where}:" in str(error)


def test_min_replicas_may_not_exceed_max_replicas(root: Path) -> None:
    write(
        root,
        "served-models/m1.yaml",
        MODEL + "    serving: {min_replicas: 3, max_replicas: 2}\n",
    )

    with pytest.raises(CatalogInvalid) as raised:
        load_catalog(root)

    (error,) = raised.value.errors
    assert "min_replicas" in str(error) and "max_replicas" in str(error)


def test_an_unknown_nested_key_is_named_by_its_full_path(root: Path) -> None:
    write(
        root,
        "served-models/m1.yaml",
        MODEL.replace("wait_s: 0.01", "wait_s: 0.01, wait: 1").replace(
            "range: [0, 1]", "range: [0, 1], chanels: 2"
        ),
    )

    with pytest.raises(CatalogInvalid) as raised:
        load_catalog(root)

    assert sorted(str(e) for e in raised.value.errors) == [
        "catalog/served-models/m1.yaml:batching.wait: unknown key `batching.wait`. "
        "Keys of batching: max_size, wait_s. Hint: did you mean `wait_s`?",
        "catalog/served-models/m1.yaml:output.chanels: unknown key "
        "`output.chanels`. Keys of output: channels, dtype, kind, range, size. "
        "Hint: did you mean `channels`?",
    ]
