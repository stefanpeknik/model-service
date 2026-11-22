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
    weights: {use: local, path: /weights/model.onnx}
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
    return load_catalog(root).entries("served-models")[model_id].value


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


def test_keys_of_todays_model_files_are_refused_with_what_fits(
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
        "this file: batching, input, output, post, pre, resources, runtime, "
        "serving, weights. Hint: remove it",
    ]


def test_threads_are_refused_where_session_options_are_not_applied(
    root: Path,
) -> None:
    runtime = "{use: onnx, apply_session_options: false, threads: 4}"
    write(root, "served-models/m1.yaml", MODEL.replace("{use: onnx}", runtime))

    with pytest.raises(CatalogInvalid) as raised:
        load_catalog(root)

    (error,) = raised.value.errors
    assert "threads has no effect with apply_session_options: false" in str(error)


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
        "catalog/served-models/m1.yaml:weights.revision: `main` is invalid: string "
        "should match pattern '^[0-9a-f]{40}$'. revision: The full 40-character "
        "commit hash to download. A branch or tag can move, so the weights would "
        "change under the same file. Hint: copy the full commit hash from the "
        "history of the repository's files on huggingface.co"
    ]
