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

A new contract adds a model declaration in `model_declarations.py` and a client
module here, listed in `CLIENTS`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from model_service.contracts.failures import BuilderFailure, FailureCode
from model_service.contracts.model_declarations import InferenceContract
from model_service.deployed_models.model_clients.base import ModelClient, ray_handle
from model_service.deployed_models.model_clients.spatial_scores import (
    SpatialScoresClient,
)


if TYPE_CHECKING:
    from model_service.contracts.model_declarations import ModelBinding
    from model_service.deployed_models.model_clients.base import HandleLookup

__all__ = ["CLIENTS", "ModelClient", "SpatialScoresClient", "client_for", "ray_handle"]


CLIENTS: dict[InferenceContract, type[ModelClient[Any]]] = {
    client.contract: client for client in (SpatialScoresClient,)
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
