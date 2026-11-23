"""Checks on what goes into and comes out of a served model, on every request.

A model returning logits instead of probabilities, the wrong shape or NaNs
fails loudly on its first batch instead of drawing a plausible heatmap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from model_service.catalog.served_models import ScoresOutput


if TYPE_CHECKING:
    from model_service.catalog.served_models import ServedModel


class ModelInputError(ValueError):
    """A tile the served model does not take."""


class ModelOutputError(ValueError):
    """An output that breaks the served model's `output:` declaration."""


def check_tile(model: ServedModel, tile: object) -> None:
    """Refuse a tile that is not what `input:` declares.

    Args:
        model: The served model.
        tile: One tile, expected uint8 `(3, tile_size, tile_size)`.

    Raises:
        ModelInputError: For another type, dtype or shape.
    """
    size = model.input.tile_size
    want = (3, size, size)
    if not isinstance(tile, np.ndarray) or tile.dtype != np.uint8 or tile.shape != want:
        got = (
            f"{tile.dtype} {tile.shape}"
            if isinstance(tile, np.ndarray)
            else type(tile).__name__
        )
        raise ModelInputError(
            f"{model.id} takes uint8 RGB tiles {want} (channels first), got {got}"
        )


def check_rows(model: ServedModel, output: np.ndarray, rows: int) -> None:
    """Refuse a runtime output without one row per tile.

    Args:
        model: The served model.
        output: The runtime's output for a batch.
        rows: Tiles in the batch.

    Raises:
        ModelOutputError: If the first axis is not the batch.
    """
    if output.ndim == 0 or len(output) != rows:
        raise ModelOutputError(
            f"{model.id} returned {output.shape} for a batch of {rows} tiles; "
            "the first axis must be the batch"
        )


def check_output(model: ServedModel, output: np.ndarray) -> np.ndarray:
    """Check one tile's output against `output:` and give it the declared shape.

    Scores may come in any shape holding `channels x size x size` values; they
    are returned as `(channels, size, size)`. An embedding is one vector for
    one summary, else one row per summary.

    Args:
        model: The served model.
        output: One tile's output after `post:`.

    Returns:
        The output, scores reshaped to `(channels, size, size)`.

    Raises:
        ModelOutputError: For a wrong shape, dtype, a non-finite value or a
            value outside the declared range.
    """
    declared = model.output
    if isinstance(declared, ScoresOutput):
        shape = (declared.channels, declared.size, declared.size)
        if output.size != np.prod(shape):
            raise _error(
                model,
                f"{output.shape} scores per tile, declared {shape}",
                "fix `output.channels` and `output.size`",
            )
        output = output.reshape(shape)
    else:
        count = len(declared.summaries)
        rank = 1 if count == 1 else 2
        if output.ndim != rank or (rank == 2 and output.shape[0] != count):
            want = "(dim,)" if count == 1 else f"({count}, dim)"
            raise _error(
                model,
                f"{output.shape} features per tile, declared {want} for "
                f"summaries {list(declared.summaries)}",
                "fix `output.summaries` or the `post:` steps",
            )
    if output.dtype != np.dtype(declared.dtype):
        raise _error(
            model,
            f"{output.dtype}, declared {declared.dtype}",
            "fix `output.dtype`",
        )
    if not np.isfinite(output).all():
        raise _error(
            model,
            "a value that is not finite",
            "check the weights and the input normalisation",
        )
    if isinstance(declared, ScoresOutput) and declared.range is not None:
        low, high = declared.range
        if output.min() < low or output.max() > high:
            value = output.min() if output.min() < low else output.max()
            raise _error(
                model,
                f"{value}, outside the declared range [{low}, {high}]",
                "a model that returns logits needs `post: [{use: sigmoid}]` "
                "or `post: [{use: softmax}]`; otherwise fix `output.range`",
            )
    return output


def _error(model: ServedModel, returned: str, hint: str) -> ModelOutputError:
    """An output error naming the model, what it returned and a fix.

    Args:
        model: The served model.
        returned: What the output was.
        hint: How to fix the model or its file.

    Returns:
        The error to raise.
    """
    return ModelOutputError(f"{model.id} returned {returned}. Hint: {hint}")
