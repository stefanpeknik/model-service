"""Slots: the places in catalog files where a component can be named."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Slot:
    """One key of a catalog document that takes components.

    Attributes:
        document: The kind of document, e.g. `served model` or `analysis`.
        key: The YAML key, e.g. `runtime` or `steps`.
    """

    document: str
    key: str

    def __str__(self) -> str:
        """Render as the YAML key a reader would look for, e.g. `post:`."""
        return f"`{self.key}:`"


WEIGHTS = Slot("served model", "weights")
RUNTIME = Slot("served model", "runtime")
PRE = Slot("served model", "pre")
POST = Slot("served model", "post")
SOURCE = Slot("analysis", "source")
STEPS = Slot("analysis", "steps")
STEP_WEIGHTS = Slot("analysis step", "weights")
SINKS = Slot("analysis", "sinks")
EMPAIA_INPUTS = Slot("EMPAIA exposure", "inputs")
EMPAIA_OUTPUTS = Slot("EMPAIA exposure", "outputs")
