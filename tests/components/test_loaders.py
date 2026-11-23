"""The `mlflow` loader names the weights it fetched by their content."""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

import pytest

from components.loaders.mlflow import Mlflow


if TYPE_CHECKING:
    from pathlib import Path


WEIGHTS = b"stand-in weights"
SHA256 = hashlib.sha256(WEIGHTS).hexdigest()


@pytest.fixture
def uri(tmp_path: Path) -> str:
    artifact = tmp_path / "artifacts" / "model.onnx"
    artifact.parent.mkdir()
    artifact.write_bytes(WEIGHTS)
    return artifact.as_uri()


def test_the_revision_after_fetch_names_the_content(uri: str, tmp_path: Path) -> None:
    loader = Mlflow.from_options({"uri": uri})

    fetched = loader.fetch(tmp_path / "cache")

    assert fetched.read_bytes() == WEIGHTS
    assert loader.revision() == f"{uri}@sha256:{SHA256}"
    assert loader.expected_revision() is None


def test_a_pinned_checksum_is_expected_and_checked(uri: str, tmp_path: Path) -> None:
    pinned = Mlflow.from_options({"uri": uri, "sha256": SHA256})
    other = Mlflow.from_options({"uri": uri, "sha256": "0" * 64})

    pinned.fetch(tmp_path / "pinned")

    assert pinned.revision() == pinned.expected_revision() == f"{uri}@sha256:{SHA256}"
    with pytest.raises(ValueError, match="sha256"):
        other.fetch(tmp_path / "other")
