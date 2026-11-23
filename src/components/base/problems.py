"""What a check found wrong with a component's use, handed to its `hint()`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Problem:
    """One failed check on a component, before it becomes a catalog error.

    Attributes:
        code: What kind of check failed, e.g. `wrong_slot`, `unknown_option`,
            `missing_option`, `invalid_option`.
        message: What's wrong, as one sentence without a final period.
        fits: What would fit instead, as one sentence without a final period.
        detail: Facts a hint may use, e.g. the option name or the slot.
    """

    code: str
    message: str
    fits: str
    detail: dict[str, Any] = field(default_factory=dict)
