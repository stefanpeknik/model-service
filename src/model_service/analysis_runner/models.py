"""Model steps at run time: call a served model's handle per tile, with failure codes."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Protocol

import numpy as np

from model_service.contracts.failures import BuilderFailure, FailureCode
from model_service.served_model.guards import ModelOutputError, check_output


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from model_service.catalog.analyses import ModelStep
    from model_service.components import Batch


class ModelHandle(Protocol):
    """A served model as the runner calls it: the host's `infer(tile)`."""

    async def infer(self, tile: np.ndarray) -> np.ndarray:
        """Infer one tile.

        Args:
            tile: uint8 `(3, tile_size, tile_size)`.

        Returns:
            The tile's scores or embedding.
        """
        ...


def model_call(
    step: ModelStep, handle: ModelHandle
) -> Callable[[Batch], Awaitable[Batch]]:
    """Turn a model step and its served model's handle into a batch function.

    Each tile is its own call, so the served model batches tiles across runs.

    Args:
        step: The model step.
        handle: The served model's handle.

    Returns:
        A function adding the step's column (`scores` or `features`, float32,
        in the declared shape) to a batch with `tile`.

    Raises:
        BuilderFailure: From the returned function: `model_failed` (retryable)
            when a call fails, `invalid_model_output` (not retryable) when an
            output breaks the served model's `output:` declaration.
    """
    model = step.model

    def invalid() -> BuilderFailure:
        return BuilderFailure(
            code=FailureCode.INVALID_MODEL_OUTPUT,
            safe_message=f"model {model.id!r} returned an invalid output",
            retryable=False,
        )

    async def infer(tile: np.ndarray) -> np.ndarray:
        try:
            output = await handle.infer(tile)
        except ModelOutputError as error:
            raise invalid() from error
        except Exception as error:
            raise BuilderFailure(
                code=FailureCode.MODEL_FAILED,
                safe_message=f"model {model.id!r} request failed",
                retryable=True,
            ) from error
        try:
            return check_output(model, np.asarray(output))
        except ModelOutputError as error:
            raise invalid() from error

    async def call(batch: Batch) -> Batch:
        tiles = batch["tile"]
        # One await per tile, so one tile in flight interleaves as today.
        if len(tiles) == 1:
            outputs = [await infer(tiles[0])]
        else:
            outputs = list(await asyncio.gather(*(infer(t) for t in tiles)))
        return batch | {step.column: np.stack(outputs).astype(np.float32, copy=False)}

    return call
