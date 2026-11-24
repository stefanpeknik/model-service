"""The repo's `catalog/` loads for the base files and every environment's patches.

This runs every load-time check the control plane runs, through the same loader and
component registry, and lists every mistake in the standard format:
`uv run pytest tests/catalog`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from model_service.catalog import CatalogInvalid, environments, load_catalog
from tests.support.paths import CATALOG, ENVIRONMENTS


if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    "environment",
    [None, *environments(ENVIRONMENTS)],
    ids=lambda e: "base" if e is None else e.name,
)
def test_the_catalog_loads(environment: Path | None) -> None:
    errors = ""
    try:
        load_catalog(CATALOG, environment)
    except CatalogInvalid as invalid:
        errors = str(invalid)
    if errors:
        pytest.fail(f"\n{errors}", pytrace=False)
