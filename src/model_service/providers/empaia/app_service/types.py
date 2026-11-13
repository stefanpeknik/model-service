"""What EMPAIA tells us about a job's slide and region of interest.

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


@dataclass(frozen=True, slots=True)
class SlideMetadata:
    """Carry slide identity, base dimensions, nanometer pixel sizes, and pyramid levels."""

    identifier: str
    width: int
    height: int
    pixel_size_nm_x: float
    pixel_size_nm_y: float
    levels: tuple[SlideLevel, ...]


@dataclass(frozen=True, slots=True)
class Rectangle:
    """Carry integer slide-space coordinates and dimensions for a region of interest."""

    x: int
    y: int
    width: int
    height: int
