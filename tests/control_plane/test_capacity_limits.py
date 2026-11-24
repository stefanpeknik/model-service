"""Scheduled-run slots per model come from its file, with PER_MODEL_LIMIT as default."""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_service.catalog import load_catalog
from model_service.control_plane.app_factory import capacity_limits
from model_service.control_plane.scheduler.targets import CapacityKey, CapacityKind
from tests.support.pipeline_catalog import model_file, write


if TYPE_CHECKING:
    from pathlib import Path


def test_a_model_file_sets_its_slots_and_others_get_the_default(
    tmp_path: Path,
) -> None:
    root = tmp_path / "catalog"
    write(
        root,
        "models/wide.yaml",
        model_file().rstrip() + "\n        serving: {max_concurrent_runs: 3}\n",
    )
    write(root, "models/plain.yaml", model_file())
    catalog = load_catalog(root)
    limits = capacity_limits(lambda: catalog, default=1)

    assert limits(CapacityKey(CapacityKind.MODEL, "wide")) == 3
    assert limits(CapacityKey(CapacityKind.MODEL, "plain")) == 1
    assert limits(CapacityKey(CapacityKind.MODEL, "removed")) == 1
    # A model-free pipeline named like a model is its own pool.
    assert limits(CapacityKey(CapacityKind.PIPELINE, "wide")) == 1
