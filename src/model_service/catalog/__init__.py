"""The catalog: served-model and analysis files, checked and resolved.

`load_catalog(root, environment)` reads a catalog directory, deep-merges the
environment's patches over the base files and resolves every document through
its document kind, or raises `CatalogInvalid` listing every mistake as
`<file>:<yaml path>: <what's wrong>. <what would fit>. Hint: <how to fix>`.
The API, the catalog test and tools all load through it. It imports no Ray,
ONNX Runtime, torch or OpenSlide.
"""

from model_service.catalog.catalog import Catalog, Entry
from model_service.catalog.context import ResolveContext
from model_service.catalog.errors import CatalogError, CatalogInvalid
from model_service.catalog.kinds import DEFAULT_KINDS, DocumentKind
from model_service.catalog.loader import environments, load_catalog


__all__ = [
    "DEFAULT_KINDS",
    "Catalog",
    "CatalogError",
    "CatalogInvalid",
    "DocumentKind",
    "Entry",
    "ResolveContext",
    "environments",
    "load_catalog",
]
