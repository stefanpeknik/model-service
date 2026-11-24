"""Read the catalog from the file the chart's catalog ConfigMap holds.

`deploy/helm/templates/model-catalog.yaml` renders `catalog.json`:
`{"models": {id: document}, "pipelines": {id: document}}`, the deployed
models and pipelines after the environment's patches. They resolve
through the catalog loader, so the control plane runs the same load-time checks as the
catalog test, and a pipeline naming a model that isn't deployed is an
error naming that model.
"""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import yaml

from model_service.catalog import Catalog, CatalogInvalid, load_catalog


CATALOG_FILE = "catalog.json"


class CatalogError(ValueError):
    """The mounted files do not form a valid catalog."""


def read_catalog(directory: Path) -> Catalog:
    """Resolve the mounted catalog.

    Args:
        directory: Where the ConfigMap is mounted.

    Returns:
        The models and pipelines, as the catalog loader resolved them.

    Raises:
        CatalogError: If the file cannot be read, or a model or
            pipeline does not resolve (the message lists every error in the
            catalog's format).
    """
    return _resolve(_read_json(directory / CATALOG_FILE))


def _read_json(path: Path) -> Any:
    """Read one JSON file of the mounted catalog.

    Args:
        path: The file.

    Returns:
        The parsed JSON.

    Raises:
        CatalogError: If it can't be read or parsed.
    """
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CatalogError(f"cannot read model catalog file {path}") from error


def _resolve(data: object) -> Catalog:
    """Resolve `catalog.json` through the catalog loader.

    The loader reads a catalog directory, so the documents are written out
    as one under a temporary directory first; errors then name
    `catalog/models/<id>.yaml`, as in the repo.

    Args:
        data: The parsed `catalog.json`: folder to `{id: document}`.

    Returns:
        The resolved catalog.

    Raises:
        CatalogError: For a malformed file, an unsafe name or an invalid catalog.
    """
    if not isinstance(data, dict) or not all(
        isinstance(documents, dict) for documents in data.values()
    ):
        raise CatalogError(f"{CATALOG_FILE} must map folders to {{id: document}}")
    with TemporaryDirectory() as temporary:
        root = Path(temporary) / "catalog"
        for folder, documents in data.items():
            for document_id, document in documents.items():
                if not _safe_name(folder) or not _safe_name(document_id):
                    raise CatalogError(
                        f"{CATALOG_FILE}: {folder}/{document_id} is not a file name"
                    )
                path = root / folder / f"{document_id}.yaml"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(yaml.safe_dump(document), encoding="utf-8")
        try:
            catalog = load_catalog(root)
        except CatalogInvalid as invalid:
            raise CatalogError(str(invalid)) from invalid
    return catalog


def _safe_name(name: object) -> bool:
    """Whether a folder or document name is safe to write as a file name.

    Args:
        name: The name from `catalog.json`.

    Returns:
        True for a non-empty string without separators or a leading dot.
    """
    return (
        isinstance(name, str)
        and bool(name)
        and "/" not in name
        and "\\" not in name
        and not name.startswith(".")
    )
