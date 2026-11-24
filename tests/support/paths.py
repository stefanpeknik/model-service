"""Where the repo keeps the catalog and the deployment environments."""

from __future__ import annotations

from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
CATALOG = REPO / "catalog"
ENVIRONMENTS = REPO / "deploy" / "helm" / "environments"
RATIONAI_TEST = ENVIRONMENTS / "rationai-test"
