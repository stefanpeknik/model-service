"""Slots: the places in catalog files where a component can be named."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Slot:
    """One key of a catalog document that takes components.

    Attributes:
        document: The kind of document, e.g. `model` or `pipeline`.
        key: The YAML key, e.g. `runtime` or `steps`.
    """

    document: str
    key: str

    def __str__(self) -> str:
        """Render as the YAML key a reader would look for, e.g. `post:`."""
        return f"`{self.key}:`"


LOADER = Slot("model", "loader")
RUNTIME = Slot("model", "runtime")
PRE = Slot("model", "pre")
POST = Slot("model", "post")
SOURCE = Slot("pipeline", "source")
STEPS = Slot("pipeline", "steps")
STEP_LOADER = Slot("pipeline step", "loader")
SINKS = Slot("pipeline", "sinks")
EMPAIA_INPUTS = Slot("EMPAIA provider settings", "inputs")
EMPAIA_OUTPUTS = Slot("EMPAIA provider settings", "outputs")
