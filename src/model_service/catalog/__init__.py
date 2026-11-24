"""The catalog: model and pipeline files, checked and resolved.

`load_catalog(root, environment)` reads a catalog directory, deep-merges the
environment's patches over the base files and resolves every model,
then every pipeline, or raises `CatalogInvalid` listing every mistake as
`<file>:<yaml path>: <what's wrong>. <what would fit>. Hint: <how to fix>`.
The control plane, the catalog test and tools all load through it. It imports no Ray,
ONNX Runtime, torch or OpenSlide.
"""

from model_service.catalog.catalog import Catalog
from model_service.catalog.context import ResolveContext
from model_service.catalog.errors import CatalogError, CatalogInvalid
from model_service.catalog.loader import environments, load_catalog


__all__ = [
    "Catalog",
    "CatalogError",
    "CatalogInvalid",
    "ResolveContext",
    "environments",
    "load_catalog",
]
