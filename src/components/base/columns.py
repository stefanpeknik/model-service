"""Columns a component reads and adds, with symbolic dimensions.

A batch is `dict[str, ndarray]`; row i of every column belongs to the same item.
A column's declared shape leaves out the batch axis. A dimension is a fixed
size, `None` for any size, the name of one of the component's options, or
`model.<path>` for a field of the served-model declaration (for example
`model.input.tile_size`), filled in once the model is known.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    import numpy as np
    from pydantic import BaseModel


type Dim = int | str | None
type Batch = dict[str, np.ndarray]

MODEL_PREFIX = "model."


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

    def resolve(self, options: BaseModel, model: object | None) -> Col:
        """Fill in option and model dimensions where they are known.

        Args:
            options: The component's validated options.
            model: The served-model declaration, or None before it is known.

        Returns:
            The column with every resolvable dimension replaced by its value.
        """
        if self.shape is None:
            return self
        dims: list[Dim] = []
        origins: list[str] = []
        for dim in self.shape:
            if isinstance(dim, str) and dim.startswith(MODEL_PREFIX):
                value = model_field(model, dim) if model is not None else None
                if value is not None:
                    origins.append(f"model {getattr(model, 'id', '')}".rstrip())
                    dim = value
            elif isinstance(dim, str) and getattr(options, dim, None) is not None:
                origins.append(f"{dim}: {getattr(options, dim)}")
                dim = getattr(options, dim)
            dims.append(dim)
        return Col(self.dtype, tuple(dims), ", ".join(dict.fromkeys(origins)))


def model_field(model: object, dim: str) -> Any:
    """Look up a `model.<path>` dimension on a served-model declaration.

    Args:
        model: The declaration; nested parts may be objects or mappings.
        dim: The symbolic dimension, e.g. `model.output.channels`.

    Returns:
        The value at the path, or None if any part is missing.
    """
    value: Any = model
    for part in dim.removeprefix(MODEL_PREFIX).split("."):
        if isinstance(value, Mapping):
            value = value.get(part)
        else:
            value = getattr(value, part, None)
        if value is None:
            return None
    return value


def rows(batch: Batch) -> int:
    """The number of rows in a batch (0 for a batch with no columns).

    Args:
        batch: Column name to array, every array the same length.

    Returns:
        The row count.
    """
    return len(next(iter(batch.values()))) if batch else 0
