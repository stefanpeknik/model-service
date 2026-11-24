"""Scheduled-run slots per served model come from its file, with PER_MODEL_LIMIT as default."""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_service.catalog import load_catalog
from model_service.control_plane.app_factory import capacity_limits
from model_service.control_plane.scheduler.targets import CapacityKey, CapacityKind
from tests.support.analysis_catalog import served_model, write


if TYPE_CHECKING:
    from pathlib import Path


def test_a_served_model_file_sets_its_slots_and_others_get_the_default(
    tmp_path: Path,
) -> None:
    root = tmp_path / "catalog"
    write(
        root,
        "served-models/wide.yaml",
        served_model().rstrip() + "\n        serving: {max_concurrent_runs: 3}\n",
    )
    write(root, "served-models/plain.yaml", served_model())
    catalog = load_catalog(root)
    limits = capacity_limits(lambda: catalog, default=1)

    assert limits(CapacityKey(CapacityKind.MODEL, "wide")) == 3
    assert limits(CapacityKey(CapacityKind.MODEL, "plain")) == 1
    assert limits(CapacityKey(CapacityKind.MODEL, "removed")) == 1
    # A model-free analysis named like a served model is its own pool.
    assert limits(CapacityKey(CapacityKind.ANALYSIS, "wide")) == 1
