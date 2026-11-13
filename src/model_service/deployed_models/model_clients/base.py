"""The client base every contract's client extends, and the Ray handle lookup."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from model_service.contracts.failures import BuilderFailure, FailureCode
from model_service.contracts.model_declarations import (
    InferenceContract,
    SpatialScoresModel,
    TileEmbeddingModel,
)


if TYPE_CHECKING:
    from collections.abc import Callable

    from ray.serve.handle import DeploymentHandle

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
            declaration: The model's declaration.
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
                    "that differs from its declaration"
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
