"""The providers this deployment composes, in order.

Adding a provider adds its `Provider` here. Kept apart from `composition.py`,
which every provider imports for the `Provider` base.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from model_service.providers.empaia.composition import PROVIDER as EMPAIA
from model_service.providers.standalone import PROVIDER as STANDALONE


if TYPE_CHECKING:
    from model_service.providers.composition import Provider


PROVIDERS: tuple[Provider, ...] = (STANDALONE, EMPAIA)
