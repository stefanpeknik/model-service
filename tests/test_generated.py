"""The committed schemas, schema lines and reference docs are what the code generates."""

from __future__ import annotations

from pathlib import Path

import pytest

from model_service.reference import stale


REPO = Path(__file__).resolve().parents[1]


def test_generated_files_are_current() -> None:
    paths = stale(REPO)
    if paths:
        listed = "\n".join(f"  {p}" for p in paths)
        pytest.fail(
            f"stale generated files:\n{listed}\n"
            "Run: PYTHONPATH=src uv run scripts/generate.py",
            pytrace=False,
        )
