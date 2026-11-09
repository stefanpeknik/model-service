"""The providers this deployment composes, in order.

Adding a provider adds its `Provider` here. Kept apart from `plugin.py`, which
every provider imports for the `Provider` base.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_service.providers.empaia.provider import PROVIDER as EMPAIA
from model_service.providers.standalone import PROVIDER as STANDALONE


if TYPE_CHECKING:
    from model_service.providers.plugin import Provider


PROVIDERS: tuple[Provider, ...] = (STANDALONE, EMPAIA)
