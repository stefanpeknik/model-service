"""Served-model files (`catalog/served-models/<id>.yaml`) and what they resolve to.

A served model is a trained model behind one generic Serve host: a loader
(`loader:`), a runtime, optional per-tile `pre:`/`post:` steps, and the
`input`, `output`, `batching`, `resources` and `serving` blocks. Its `digest`
identifies what it computes; the host reports it from `describe()`.
"""

from model_service.catalog.served_models.declaration import ServedModel
from model_service.catalog.served_models.document import (
    FOLDER,
    build_served_model,
    resolve_served_model,
    served_model_schema,
)
from model_service.catalog.served_models.fields import (
    Batching,
    EmbeddingOutput,
    Input,
    Output,
    Resources,
    ScoresOutput,
    Serving,
)


__all__ = [
    "FOLDER",
    "Batching",
    "EmbeddingOutput",
    "Input",
    "Output",
    "Resources",
    "ScoresOutput",
    "ServedModel",
    "Serving",
    "build_served_model",
    "resolve_served_model",
    "served_model_schema",
]
