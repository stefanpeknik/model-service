"""The host reads back the served model Helm hands it in `user_config`."""

from __future__ import annotations

from pathlib import Path

from model_service.catalog import load_catalog
from model_service.served_model.deployment import DEFAULT_CACHE_ROOT, host_config


REPO = Path(__file__).resolve().parents[2]
CATALOG = REPO / "catalog"


def test_the_host_reads_back_the_model_the_catalog_resolved() -> None:
    entry = load_catalog(CATALOG).served_models["prostate-classifier-1"]
    user_config = {"id": entry.id, "declaration": entry.declaration}

    model, cache_root = host_config(user_config | {"cache_root": "/cache"})
    _, default_root = host_config(user_config)

    assert (model.id, model.digest) == ("prostate-classifier-1", entry.digest)
    assert (cache_root, default_root) == (Path("/cache"), Path(DEFAULT_CACHE_ROOT))
