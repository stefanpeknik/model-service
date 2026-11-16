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
module here, listed in `CLIENTS` in registry.py; see `docs/guides/adding-builders.md`.
"""
