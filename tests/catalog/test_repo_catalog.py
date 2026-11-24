"""The repo's `catalog/` loads for the base files and every environment.

This runs every load-time check the control plane runs, through the same loader and
component registry, and lists every mistake in the standard format:
`uv run pytest tests/catalog`.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from model_service.catalog import CatalogInvalid, environments, load_catalog


CATALOG = Path(__file__).resolve().parents[2] / "catalog"


@pytest.mark.parametrize("environment", [None, *environments(CATALOG)])
def test_the_catalog_loads(environment: str | None) -> None:
    errors = ""
    try:
        load_catalog(CATALOG, environment)
    except CatalogInvalid as invalid:
        errors = str(invalid)
    if errors:
        pytest.fail(f"\n{errors}", pytrace=False)
