"""Columns a component reads and adds.

A batch is `dict[str, ndarray]`; row i of every column belongs to the same item.
A column's declared shape leaves out the batch axis. A dimension is a fixed
size, `None` for any size, or the name of one of the component's options,
filled in from the option's value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy as np
    from pydantic import BaseModel


type Dim = int | str | None
type Batch = dict[str, np.ndarray]


@dataclass(frozen=True)
class Col:
    """A column's per-row dtype and shape.

    Attributes:
        dtype: The NumPy dtype name, e.g. `uint8` or `float32`.
        shape: The dimensions without the batch axis; `None` means any rank.
        origin: Where resolved dimensions came from, for error messages.
    """

    dtype: str
    shape: tuple[Dim, ...] | None = ()
    origin: str = field(default="", compare=False)

    def __str__(self) -> str:
        """Render as `dtype[d0,d1,...]`, with `?` for unknown dimensions."""
        if self.shape is None:
            return f"{self.dtype}[...]"
        dims = ",".join("?" if d is None else str(d) for d in self.shape)
        return f"{self.dtype}[{dims}]" if self.shape else self.dtype

    def fits(self, other: Col) -> bool:
        """Whether a column declared as `other` satisfies this declaration.

        Args:
            other: The column on offer.

        Returns:
            True when dtypes and ranks match and every dimension that is a
            number on both sides is equal.
        """
        if self.shape is None or other.shape is None:
            return self.dtype == other.dtype
        return (
            self.dtype == other.dtype
            and len(self.shape) == len(other.shape)
            and all(
                a == b or not (isinstance(a, int) and isinstance(b, int))
                for a, b in zip(self.shape, other.shape, strict=True)
            )
        )

    def resolve(self, options: BaseModel) -> Col:
        """Fill in the dimensions named after options.

        Args:
            options: The component's validated options.

        Returns:
            The column with each option-named dimension replaced by its value.
        """
        if self.shape is None:
            return self
        dims: list[Dim] = []
        origins: list[str] = []
        for dim in self.shape:
            if isinstance(dim, str) and getattr(options, dim, None) is not None:
                origins.append(f"{dim}: {getattr(options, dim)}")
                dim = getattr(options, dim)
            dims.append(dim)
        return Col(self.dtype, tuple(dims), ", ".join(dict.fromkeys(origins)))


def resolved(columns: Mapping[str, Col], options: BaseModel) -> dict[str, Col]:
    """Columns with their option-named dimensions filled in.

    Args:
        columns: A component's `reads`, `adds` or `frame`.
        options: The component's validated options.

    Returns:
        The columns, each resolved against `options`.
    """
    return {name: col.resolve(options) for name, col in columns.items()}


def rows(batch: Batch) -> int:
    """The number of rows in a batch (0 for a batch with no columns).

    Args:
        batch: Column name to array, every array the same length.

    Returns:
        The row count.
    """
    return len(next(iter(batch.values()))) if batch else 0
