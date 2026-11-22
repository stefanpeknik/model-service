"""Golden references: what a deployment answered, and how a later one is compared.

A reference set is a directory:

- `raw/<model>/tiles.npy`: the uint8 RGB tiles sent, shaped (n, size, size, 3);
- `raw/<model>/replies.npz`: each reply decoded to an array, keyed `tile_<i>`
  (and `unpooled_<i>` for the embedding route without pooling);
- `raw/<model>/case.json`: the slides, levels and positions the tiles came from;
- `runs/<analysis>/<slide>.<ext>`: the output of one synchronous run per slide.

The repo keeps only `manifest.json` (file hashes and provenance) next to this
module; the files live on the deployment's shared storage and in a local copy.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np


if TYPE_CHECKING:
    from numpy.typing import NDArray

MANIFEST = Path(__file__).with_name("manifest.json")

ReplyKind = Literal["score", "score_map", "embedding"]


@dataclass(frozen=True, slots=True)
class RawCase:
    """A served model's raw route, as the references exercise it.

    Attributes:
        model: The served model's ID, also its route prefix.
        kind: How its reply is decoded: a JSON float, an LZ4 score map or an
            LZ4 embedding with a shape header.
        tile_size: Edge of the square RGB tile it takes.
        mpp: Microns per pixel the tiles are read at.
        dtype: The score map's element type (score maps only).
    """

    model: str
    kind: ReplyKind
    tile_size: int
    mpp: float
    dtype: str = "float32"


@dataclass(frozen=True, slots=True)
class RunCase:
    """A v1 analysis, recorded through today's builder and model route.

    Attributes:
        analysis: The analysis ID it becomes.
        builder: Today's builder.
        model: Today's model.
        extension: The output file's extension.
    """

    analysis: str
    builder: str
    model: str
    extension: str


RAW_CASES = {
    case.model: case
    for case in (
        RawCase("prostate-classifier-1", "score", 512, 0.468),
        RawCase("episeg-1", "score_map", 1024, 0.468, dtype="float16"),
        RawCase("virchow2", "embedding", 224, 0.5),
        RawCase("prov-gigapath", "embedding", 224, 0.5),
    )
}

RUN_CASES = {
    case.analysis: case
    for case in (
        RunCase("prostate-heatmap", "heatmap", "prostate-classifier-1", "npz"),
        RunCase("episeg-heatmap", "heatmap", "episeg-1", "npz"),
        RunCase("virchow2-embeddings", "tile-embeddings", "virchow2", "h5"),
        RunCase("prov-gigapath-embeddings", "tile-embeddings", "prov-gigapath", "h5"),
    )
}

# Scores within this of the reference count as unchanged on a GPU.
SCORE_ATOL = 1e-3
# Per-tile cosine similarity an embedding must reach.
MIN_COSINE = 0.9999


def sha256(path: Path) -> str:
    """Hash a file.

    Args:
        path: The file.

    Returns:
        Its SHA-256 as hex.
    """
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest() -> dict[str, object]:
    """Read the committed manifest.

    Returns:
        The manifest, or an empty one when none is committed yet.
    """
    if not MANIFEST.exists():
        return {}
    return json.loads(MANIFEST.read_text())


def check_hashes(root: Path, manifest: dict[str, object]) -> list[str]:
    """Compare a local reference set with the manifest's hashes.

    Args:
        root: The local reference directory.
        manifest: The committed manifest.

    Returns:
        One line per missing or changed file; empty when the set is intact.
    """
    problems = []
    files = manifest.get("files", {})
    assert isinstance(files, dict)
    for name, expected in files.items():
        path = root / name
        if not path.exists():
            problems.append(f"{name}: missing")
        elif sha256(path) != expected:
            problems.append(f"{name}: hash differs from the manifest")
    return problems


def cosine_rows(
    actual: NDArray[np.floating], expected: NDArray[np.floating]
) -> NDArray[np.float64]:
    """Cosine similarity of matching rows.

    Args:
        actual: Rows to check, shaped (n, d) or flattened per row.
        expected: Reference rows of the same shape.

    Returns:
        One similarity per row.
    """
    a = actual.reshape(len(actual), -1).astype(np.float64)
    e = expected.reshape(len(expected), -1).astype(np.float64)
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(e, axis=1)
    return np.einsum("ij,ij->i", a, e) / np.where(norms == 0, 1.0, norms)


def compare_reply(
    kind: ReplyKind, actual: NDArray[np.generic], expected: NDArray[np.generic]
) -> str | None:
    """Compare one decoded raw reply with its reference.

    Args:
        kind: How the reply was decoded.
        actual: The reply now.
        expected: The recorded reply.

    Returns:
        Why they differ, or None when they match within the live tolerances.
    """
    if actual.shape != expected.shape:
        return f"shape {actual.shape} != {expected.shape}"
    if kind == "embedding":
        similarity = float(cosine_rows(actual[None], expected[None])[0])
        if similarity < MIN_COSINE:
            return f"cosine {similarity:.6f} < {MIN_COSINE}"
        return None
    difference = float(
        np.max(np.abs(actual.astype(np.float64) - expected.astype(np.float64)))
    )
    if difference > SCORE_ATOL:
        return f"max abs difference {difference:.2e} > {SCORE_ATOL}"
    return None


def compare_score_grids(actual: Path, expected: Path) -> list[str]:
    """Compare two score grid files with the live tolerances.

    Args:
        actual: The grid a deployment wrote now.
        expected: The reference grid.

    Returns:
        What differs; empty when they match.
    """
    with np.load(actual) as a, np.load(expected) as e:
        problems = [
            f"{key} differs"
            for key in ("level", "level_size", "cell_size", "mpp")
            if not np.array_equal(a[key], e[key])
        ]
        if not np.array_equal(a["scored"], e["scored"]):
            problems.append("scored mask differs")
        if a["values"].shape != e["values"].shape:
            problems.append(f"values shape {a['values'].shape} != {e['values'].shape}")
        else:
            difference = float(np.max(np.abs(a["values"] - e["values"]), initial=0.0))
            if difference > SCORE_ATOL:
                problems.append(f"values differ by up to {difference:.2e}")
    return problems


def compare_embedding_tables(actual: Path, expected: Path) -> list[str]:
    """Compare two embedding tables (HDF5) with the live tolerances.

    Args:
        actual: The table a deployment wrote now.
        expected: The reference table.

    Returns:
        What differs; empty when they match.
    """
    import h5py

    with h5py.File(actual, "r") as a, h5py.File(expected, "r") as e:
        problems = [
            f"attribute {key} differs"
            for key in sorted(set(a.attrs) | set(e.attrs))
            if not np.array_equal(
                np.asarray(a.attrs.get(key)), np.asarray(e.attrs.get(key))
            )
        ]
        if not np.array_equal(a["coordinates"][()], e["coordinates"][()]):
            problems.append("coordinates differ")
            return problems
        if "features" in a or "features" in e:
            if a["features"].shape != e["features"].shape:
                problems.append(
                    f"features shape {a['features'].shape} != {e['features'].shape}"
                )
            elif len(a["features"]):
                worst = float(cosine_rows(a["features"][()], e["features"][()]).min())
                if worst < MIN_COSINE:
                    problems.append(f"lowest row cosine {worst:.6f} < {MIN_COSINE}")
    return problems
