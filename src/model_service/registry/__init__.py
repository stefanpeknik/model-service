"""Finds the components in `components/` and enforces where each may be used.

- scan.py: `Registry.scan()` imports the component folders and checks each
  component sits in a folder its kind belongs to (`FOLDERS`);
  `default_registry()` is this repo's scan, cached per process.
- empaia.py: picks the EMPAIA converter or publisher for a declared type.

The rules live here, not in `components/`, so that package holds only the
pieces and the classes they subclass.
"""

from model_service.registry.scan import (
    FOLDERS,
    Registry,
    RegistryError,
    default_registry,
)


__all__ = ["FOLDERS", "Registry", "RegistryError", "default_registry"]
