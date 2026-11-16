"""The client registry: which client serves each inference contract.

A new contract's client is added to `CLIENTS`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from model_service.contracts.failures import BuilderFailure, FailureCode
from model_service.contracts.model_declarations import InferenceContract
from model_service.models.clients.base import ModelClient, ray_handle
from model_service.models.clients.spatial_scores import SpatialScoresClient
from model_service.models.clients.tile_embedding import TileEmbeddingClient


if TYPE_CHECKING:
    from model_service.contracts.model_declarations import ModelBinding
    from model_service.models.clients.base import HandleLookup


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

    Raises:
        BuilderFailure: No client serves the contract (not retryable).
    """
    try:
        client = CLIENTS[declaration.contract]
    except KeyError as error:
        raise BuilderFailure(
            code=FailureCode.MODEL_CONTRACT_MISMATCH,
            safe_message=f"no model client serves {declaration.contract!r}",
            retryable=False,
        ) from error
    return client(declaration, handles)
