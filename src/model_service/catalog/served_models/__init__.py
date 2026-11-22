"""Served-model files (`catalog/served-models/<id>.yaml`) and what they resolve to.

A served model is a trained model behind one generic Serve host: a loader
(`weights:`), a runtime, optional per-tile `pre:`/`post:` steps, and the
`input`, `output`, `batching`, `resources` and `serving` blocks. Its `digest`
identifies what it computes; the host reports it from `describe()`.
"""

from model_service.catalog.served_models.declaration import ServedModel
from model_service.catalog.served_models.fields import (
    Batching,
    EmbeddingOutput,
    Input,
    Output,
    Resources,
    ScoresOutput,
    Serving,
)
from model_service.catalog.served_models.kind import (
    FOLDER,
    ServedModels,
    resolve_served_model,
)
from model_service.catalog.served_models.legacy import legacy_config


__all__ = [
    "FOLDER",
    "Batching",
    "EmbeddingOutput",
    "Input",
    "Output",
    "Resources",
    "ScoresOutput",
    "ServedModel",
    "ServedModels",
    "Serving",
    "legacy_config",
    "resolve_served_model",
]
