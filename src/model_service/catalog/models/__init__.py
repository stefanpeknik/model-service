"""Model files (`catalog/models/<id>.yaml`) and what they resolve to.

A model is a trained model behind one generic Serve host: a loader
(`loader:`), a runtime, optional per-tile `pre:`/`post:` steps, and the
`input`, `output`, `batching`, `resources` and `serving` blocks. Its `digest`
identifies what it computes; the host reports it from `describe()`.
"""

from model_service.catalog.models.declaration import Model
from model_service.catalog.models.document import (
    FOLDER,
    build_model,
    model_schema,
    resolve_model,
)
from model_service.catalog.models.fields import (
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
    "Model",
    "Output",
    "Resources",
    "ScoresOutput",
    "Serving",
    "build_model",
    "model_schema",
    "resolve_model",
]
