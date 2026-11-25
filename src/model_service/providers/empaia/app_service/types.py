"""What EMPAIA tells us about a job's slide.

The gateway reads these from the App Service; materialization and publication
use them to place inputs and outputs on the slide.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True, slots=True)
class DirectAccess:
    """Describe a shared input file and the provider's direct-access support flag."""

    storage_path: Path
    supported: bool


@dataclass(frozen=True, slots=True)
class SlideLevel:
    """Describe the index and pixel dimensions of one slide pyramid level."""

    index: int
    width: int
    height: int
