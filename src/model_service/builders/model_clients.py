"""Call a model deployment the way its inference contract defines, one client per contract.

A builder never talks to Ray itself: the application creates the client for the
model's contract, checks the running model against its catalog declaration, and
hands the builder the verified client as `context.model`. A client calls the
model through its Ray Serve application handle (tiles stay numpy arrays inside
Ray) and checks every answer, so all builders of a contract share the same
failure handling:

- a running model whose `get_config()` differs from the catalog:
  `model_contract_mismatch`, not retried;
- a failed model call: `model_failed`, retried;
- an answer of the wrong shape, type or with non-finite values:
  `invalid_model_output`, not retried.

A new contract adds a model declaration in `contracts.py` and a client here,
listed in `CLIENTS`; see `docs/guides/adding-builders.md`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import ValidationError

from model_service.builders.contracts import (
    BuilderFailure,
    FailureCode,
    InferenceContract,
    SpatialScoresModel,
    TileEmbeddingModel,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from numpy.typing import NDArray
    from ray.serve.handle import DeploymentHandle

    from model_service.builders.contracts import ModelBinding

    type HandleLookup = Callable[[str], DeploymentHandle[Any]]


def ray_handle(model_id: str) -> DeploymentHandle[Any]:
    """Look up a model's Ray Serve application handle.

    Args:
        model_id: Model id, which is also the application name.

    Returns:
        The application's handle.
    """
    from ray import serve

    return serve.get_app_handle(model_id)


class ModelClient[M: SpatialScoresModel | TileEmbeddingModel](ABC):
    """A model deployment of one contract, as a builder sees it."""

    contract: ClassVar[InferenceContract]
    # Methods a model implementation of this contract must provide.
    methods: ClassVar[tuple[str, ...]]

    def __init__(self, declaration: M, handles: HandleLookup = ray_handle) -> None:
        """Bind the client to a model deployment; the handle is looked up on first use.

        Args:
            declaration: The model's catalog declaration.
            handles: Looks up a deployment handle by model id; replaceable in tests.
        """
        self.declaration = declaration
        self._handles = handles
        self._handle: DeploymentHandle[Any] | None = None

    async def verify(self) -> None:
        """Fail when the running model's configuration differs from its declaration.

        Raises:
            BuilderFailure: The model cannot be reached (retryable), or its running
                configuration is invalid or differs (not retryable).
        """
        values = await self._call("get_config")
        if not isinstance(values, dict) or not self._matches(values):
            raise BuilderFailure(
                code=FailureCode.MODEL_CONTRACT_MISMATCH,
                safe_message=(
                    f"model {self.declaration.id!r} is running with a configuration "
                    "that differs from its catalog declaration"
                ),
                retryable=False,
            )

    @abstractmethod
    def _matches(self, running: dict[str, object]) -> bool:
        """Compare the running model's `get_config()` with the declaration.

        Args:
            running: What the model's `get_config()` returned.

        Returns:
            Whether the running model is the declared one.
        """
        ...

    async def _call(self, method: str, *args: object) -> object:
        """Call a method of the model deployment.

        Args:
            method: Deployment method name.
            *args: Arguments passed to it.

        Returns:
            What the method returned.

        Raises:
            BuilderFailure: The call failed (retryable).
        """
        try:
            if self._handle is None:
                self._handle = self._handles(self.declaration.id)
            return await getattr(self._handle, method).remote(*args)
        except Exception as error:
            raise BuilderFailure(
                code=FailureCode.MODEL_FAILED,
                safe_message=f"model {self.declaration.id!r} request failed",
                retryable=True,
            ) from error

    def _invalid_output(self) -> BuilderFailure:
        """Build the failure for an answer that breaks the contract.

        Returns:
            A non-retryable `invalid_model_output` failure.
        """
        return BuilderFailure(
            code=FailureCode.INVALID_MODEL_OUTPUT,
            safe_message=f"model {self.declaration.id!r} returned an invalid output",
            retryable=False,
        )


class SpatialScoresClient(ModelClient[SpatialScoresModel]):
    contract = InferenceContract.SPATIAL_SCORES
    methods = ("get_config", "predict")

    def _matches(self, running: dict[str, object]) -> bool:
        """Compare the running geometry with the declaration.

        Args:
            running: `tile_size`, `output_tile_size`, `n_channels` and `mpp`.

        Returns:
            Whether every value equals the declaration's.
        """
        try:
            model = SpatialScoresModel.model_validate(
                {"id": self.declaration.id, **running}
            )
        except ValidationError:
            return False
        return model == self.declaration

    async def predict(self, tile: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Score one tile.

        Args:
            tile: CHW uint8 tile of the declared `tile_size`.

        Returns:
            Finite scores shaped `(n_channels, output_tile_size, output_tile_size)`.

        Raises:
            BuilderFailure: The call failed (retryable) or returned the wrong
                number of values or a non-finite one (not retryable).
        """
        import numpy as np

        model = self.declaration
        shape = (model.n_channels, model.output_tile_size, model.output_tile_size)
        try:
            scores = np.asarray(await self._call("predict", tile), dtype=np.float32)
        except (TypeError, ValueError) as error:
            raise self._invalid_output() from error
        if scores.size != np.prod(shape) or not np.isfinite(scores).all():
            raise self._invalid_output()
        return scores.reshape(shape)


class TileEmbeddingClient(ModelClient[TileEmbeddingModel]):
    contract = InferenceContract.TILE_EMBEDDING
    methods = ("get_config", "embed")

    def _matches(self, running: dict[str, object]) -> bool:
        """Compare the running tile size and summaries with the declaration.

        Args:
            running: At least `tile_size` and `summaries`.

        Returns:
            Whether both equal the declaration's.
        """
        return running.get("tile_size") == self.declaration.tile_size and running.get(
            "summaries"
        ) == list(self.declaration.summaries)

    async def embed(self, tile: NDArray[np.uint8]) -> NDArray[np.float32]:
        """Embed one tile.

        Args:
            tile: CHW uint8 RGB tile of the declared `tile_size`.

        Returns:
            A `(dim,)` vector for one summary, else `(len(summaries), dim)`.

        Raises:
            BuilderFailure: The call failed (retryable) or did not return finite,
                non-empty float32 features of that shape (not retryable).
        """
        import numpy as np

        features = await self._call("embed", tile)
        summaries = len(self.declaration.summaries)
        if (
            not isinstance(features, np.ndarray)
            or features.dtype != np.float32
            or features.ndim != (1 if summaries == 1 else 2)
            or (summaries > 1 and features.shape[0] != summaries)
            or features.shape[-1] == 0
            or not np.isfinite(features).all()
        ):
            raise self._invalid_output()
        return features


CLIENTS: dict[InferenceContract, type[ModelClient[Any]]] = {
    client.contract: client for client in (SpatialScoresClient, TileEmbeddingClient)
}


def client_for(
    declaration: ModelBinding, handles: HandleLookup = ray_handle
) -> ModelClient[Any]:
    """Create the client for a model's contract.

    Args:
        declaration: The model's catalog declaration.
        handles: Looks up a deployment handle by model id.

    Returns:
        The client of the model's contract, not yet verified.
    """
    return CLIENTS[declaration.contract](declaration, handles)
