"""Model steps at run time: a served model's handle per tile, with failure codes.

`RayServedModel` is the handle on the cluster: the served model's Serve
application, called through `infer` and `describe`.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np

from model_service.contracts.failures import FailureCode, RunFailure
from model_service.served_model.guards import ModelOutputError, check_output


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from components import Batch
    from model_service.catalog.analyses import ModelStep


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


class DescribedModel(ModelHandle, Protocol):
    """A served model's handle that also says what it runs: the host's `describe()`."""

    async def describe(self) -> dict[str, Any]:
        """Report what the replica runs.

        Returns:
            At least `digest` (the declaration digest), `weights_revision` and
            `execution_providers`, which the run's result records.
        """
        ...


class RayServedModel:
    """A served model's Ray Serve application handle, as the runner calls it."""

    def __init__(self, model_id: str) -> None:
        """Look up the application named after the served model.

        Args:
            model_id: The served-model ID, also its application's name.
        """
        from ray import serve

        self._handle = serve.get_app_handle(model_id)

    async def infer(self, tile: np.ndarray) -> np.ndarray:
        """Infer one tile on the served model.

        Args:
            tile: uint8 `(3, tile_size, tile_size)`.

        Returns:
            The tile's output.
        """
        return cast("np.ndarray", await self._handle.infer.remote(tile))

    async def describe(self) -> dict[str, Any]:
        """Ask the served model what it runs.

        Returns:
            Its `describe()` reply.
        """
        return cast("dict[str, Any]", await self._handle.describe.remote())


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
        RunFailure: From the returned function: `model_failed` (retryable)
            when a call fails, `invalid_model_output` (not retryable) when an
            output breaks the served model's `output:` declaration.
    """
    model = step.model

    def invalid() -> RunFailure:
        """The failure of an output that breaks the served model's declaration.

        Returns:
            `invalid_model_output`, not retryable.
        """
        return RunFailure(
            code=FailureCode.INVALID_MODEL_OUTPUT,
            safe_message=f"model {model.id!r} returned an invalid output",
            retryable=False,
        )

    async def infer(tile: np.ndarray) -> np.ndarray:
        """Infer one tile and check the output against the declaration.

        Args:
            tile: uint8 `(3, tile_size, tile_size)`.

        Returns:
            The checked output.

        Raises:
            RunFailure: `model_failed` (retryable) when the call fails,
                `invalid_model_output` when the output breaks the declaration.
        """
        try:
            output = await handle.infer(tile)
        except ModelOutputError as error:
            raise invalid() from error
        except Exception as error:
            raise RunFailure(
                code=FailureCode.MODEL_FAILED,
                safe_message=f"model {model.id!r} request failed",
                retryable=True,
            ) from error
        try:
            return check_output(model, np.asarray(output))
        except ModelOutputError as error:
            raise invalid() from error

    async def call(batch: Batch) -> Batch:
        """Infer every tile of a batch, one call each, and add the step's column.

        Args:
            batch: Rows with `tile`.

        Returns:
            The batch with the step's column, float32.
        """
        tiles = batch["tile"]
        # A single tile skips gather, so its call is awaited directly.
        if len(tiles) == 1:
            outputs = [await infer(tiles[0])]
        else:
            outputs = list(await asyncio.gather(*(infer(t) for t in tiles)))
        return batch | {step.column: np.stack(outputs).astype(np.float32, copy=False)}

    return call
