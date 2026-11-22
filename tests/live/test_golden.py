"""Compare a running deployment with the golden references (`pytest -m live`).

Environment:

- `MODEL_SERVICE_SERVE_URL`: Ray Serve base URL (raw routes), e.g. a
  port-forward of `model-service-serve-svc` to `http://localhost:18000`;
- `MODEL_SERVICE_API_URL`: the API, e.g. `http://localhost:18080`;
- `MODEL_SERVICE_GOLDEN_DIR`: the local reference set (default `.cache/golden/refs`);
- `MODEL_SERVICE_LIVE_NAMESPACE`: where run outputs are fetched from with
  `oc exec` (default `rationai-infra-test-ns`);
- `MODEL_SERVICE_LIVE_MODELS` / `MODEL_SERVICE_LIVE_RUNS`: comma-separated
  subsets to check (default: all recorded).

Runs go through `/v1/analyses/{id}/run`.
"""

from __future__ import annotations

import os
import subprocess
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from tests.live.golden import (
    RAW_CASES,
    RUN_CASES,
    check_hashes,
    compare_embedding_tables,
    compare_reply,
    compare_score_grids,
    load_manifest,
)


pytestmark = pytest.mark.live

GOLDEN = Path(os.environ.get("MODEL_SERVICE_GOLDEN_DIR", ".cache/golden/refs"))
SHARED_ROOT = "/mnt/shared/golden"


def selected(variable: str, recorded: list[str]) -> list[str]:
    """Pick the cases to check.

    Args:
        variable: Environment variable naming a comma-separated subset.
        recorded: Cases with references.

    Returns:
        The chosen cases.
    """
    chosen = os.environ.get(variable)
    return [case for case in recorded if not chosen or case in chosen.split(",")]


def recorded(kind: str) -> list[str]:
    """List the cases a local reference set holds.

    Args:
        kind: `raw` or `runs`.

    Returns:
        Case names.
    """
    folder = GOLDEN / kind
    return sorted(path.name for path in folder.iterdir()) if folder.exists() else []


def require(variable: str) -> str:
    """Read a required environment variable or skip.

    Args:
        variable: Its name.

    Returns:
        Its value.
    """
    value = os.environ.get(variable)
    if not value:
        pytest.skip(f"{variable} is not set")
    return value


def test_reference_set_matches_manifest() -> None:
    """The local reference set is the one the manifest describes."""
    manifest = load_manifest()
    if not manifest:
        pytest.skip("no manifest committed")
    assert check_hashes(GOLDEN, manifest) == []


@pytest.mark.parametrize(
    "model", selected("MODEL_SERVICE_LIVE_MODELS", recorded("raw"))
)
def test_raw_route_matches_references(model: str) -> None:
    """Every recorded tile gets the recorded reply within the live tolerances."""
    import httpx

    from tests.live.generate_references import call_raw

    case = RAW_CASES[model]
    tiles = np.load(GOLDEN / "raw" / model / "tiles.npy")
    with np.load(GOLDEN / "raw" / model / "replies.npz") as replies:
        expected = {key: replies[key] for key in replies}
    problems = []
    with httpx.Client(
        base_url=require("MODEL_SERVICE_SERVE_URL"), timeout=1800
    ) as client:
        for key, reference in expected.items():
            pooled = key.startswith("tile_")
            tile = tiles[int(key.rsplit("_", 1)[1])]
            actual = call_raw(client, case, tile, pooled=pooled)
            if problem := compare_reply(case.kind, actual, reference):
                problems.append(f"{key}: {problem}")
    assert problems == []


@pytest.mark.parametrize(
    "analysis", selected("MODEL_SERVICE_LIVE_RUNS", recorded("runs"))
)
def test_run_matches_references(analysis: str) -> None:
    """A synchronous run on each reference slide writes the recorded output."""
    import httpx

    case = RUN_CASES[analysis]
    namespace = os.environ.get("MODEL_SERVICE_LIVE_NAMESPACE", "rationai-infra-test-ns")
    prefix = f"live/{uuid.uuid4().hex[:8]}"
    problems = []
    with httpx.Client(
        base_url=require("MODEL_SERVICE_API_URL"), timeout=7300
    ) as client:
        for reference in sorted((GOLDEN / "runs" / analysis).iterdir()):
            output = f"{prefix}/{analysis}/{reference.name}"
            response = client.post(
                f"/v1/analyses/{case.analysis}/run",
                json={
                    "inputs": {
                        "slide": {
                            "kind": "shared_path",
                            "path": f"slides/{reference.stem}.mrxs",
                        }
                    },
                    "input_access": {
                        "kind": "shared_storage",
                        "state": {"root": SHARED_ROOT},
                    },
                    "output_path": output,
                    "deadline_at": (datetime.now(UTC) + timedelta(hours=2)).isoformat(),
                },
            )
            assert response.status_code == 200, response.text
            local = Path(".cache/golden/live") / output
            local.parent.mkdir(parents=True, exist_ok=True)
            with local.open("wb") as file:
                subprocess.run(
                    [
                        "oc",
                        "-n",
                        namespace,
                        "exec",
                        "deploy/model-service-api",
                        "--",
                        "cat",
                        f"{SHARED_ROOT}/{output}",
                    ],
                    stdout=file,
                    check=True,
                )
            compare = (
                compare_score_grids
                if case.extension == "npz"
                else compare_embedding_tables
            )
            problems += [f"{reference.name}: {p}" for p in compare(local, reference)]
    assert problems == []
