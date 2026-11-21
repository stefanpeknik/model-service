"""Pytest wiring shared by every test: live tests run only when asked for."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest


if TYPE_CHECKING:
    from collections.abc import Sequence


def pytest_collection_modifyitems(
    config: pytest.Config, items: Sequence[pytest.Item]
) -> None:
    """Skip `live` tests unless the run selects them with `-m live`.

    Args:
        config: The pytest configuration, holding the `-m` expression.
        items: The collected tests.
    """
    if "live" in (config.option.markexpr or ""):
        return
    skip = pytest.mark.skip(reason="live test; select it with `-m live`")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)
