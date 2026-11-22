"""Read the model catalog from the files the chart's catalog ConfigMap holds.

`deploy/helm/rayservice/templates/model-catalog.yaml` renders three files:

- `catalog.json`: `{"served-models": {id: document}}`, the deployed served
  models after the environment's patches. They resolve through the catalog
  loader, so the API runs the same load-time checks as the catalog test;
- `model-catalog.json`: the old model files of models not yet migrated
  (transitional);
- `provider-blocks.json`: `{id: {block: value}}`, the served models' provider
  blocks, e.g. `empaia` (transitional).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import yaml

from model_service.api.model_catalog import (
    CatalogError,
    ModelCatalog,
    served_model_binding,
)
from model_service.catalog import CatalogInvalid, load_catalog
from model_service.catalog.served_models import FOLDER


if TYPE_CHECKING:
    from collections.abc import Collection

    from model_service.catalog.served_models import ServedModel
    from model_service.contracts.model_declarations import ModelBinding


logger = logging.getLogger(__name__)

CATALOG_FILE = "catalog.json"
LEGACY_FILE = "model-catalog.json"
BLOCKS_FILE = "provider-blocks.json"


def read_catalog(
    directory: Path, provider_blocks: Collection[str] = ()
) -> ModelCatalog:
    """Build the model catalog from the mounted catalog files.

    Served models come first. An old model file with a served model's ID is
    logged and left out; other old model files keep today's rules (see
    `ModelCatalog.from_documents`).

    Args:
        directory: Where the ConfigMap is mounted.
        provider_blocks: The top-level keys providers read.

    Returns:
        The catalog.

    Raises:
        CatalogError: If a file cannot be read, a served model does not
            resolve (the message lists every error in the catalog's format),
            or a provider block is one no provider takes.
    """
    served = _served_models(_read_json(directory / CATALOG_FILE))
    served_ids = {model.id for model in served}
    blocks = _provider_blocks(
        _read_json(directory / BLOCKS_FILE), served_ids, provider_blocks
    )
    legacy = ModelCatalog.from_file(directory / LEGACY_FILE, provider_blocks)
    models: list[ModelBinding] = [served_model_binding(model) for model in served]
    for model in legacy:
        if model.id in served_ids:
            logger.error(
                "model file %r is left out: a served model has its ID", model.id
            )
            continue
        models.append(model)
    for key in provider_blocks:
        old = legacy.provider_block(key)
        blocks[key] = {
            **{i: b for i, b in old.items() if i not in served_ids},
            **blocks[key],
        }
    return ModelCatalog(models, blocks)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise CatalogError(f"cannot read model catalog file {path}") from error


def _served_models(data: object) -> list[ServedModel]:
    """Resolve `catalog.json` through the catalog loader.

    The loader reads a catalog directory, so the documents are written out
    as one under a temporary directory first; errors then name
    `catalog/served-models/<id>.yaml`, as in the repo.
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
    return [entry.value for entry in catalog.entries(FOLDER).values()]


def _safe_name(name: object) -> bool:
    return (
        isinstance(name, str)
        and bool(name)
        and "/" not in name
        and "\\" not in name
        and not name.startswith(".")
    )


def _provider_blocks(
    data: object, served_ids: set[str], provider_blocks: Collection[str]
) -> dict[str, dict[str, object]]:
    """Check `provider-blocks.json` and regroup it by block key, then model ID."""
    if not isinstance(data, dict):
        raise CatalogError(f"{BLOCKS_FILE} must map model IDs to their blocks")
    blocks: dict[str, dict[str, object]] = {key: {} for key in provider_blocks}
    for model_id, model_blocks in data.items():
        if model_id not in served_ids:
            raise CatalogError(f"{BLOCKS_FILE}: no served model {model_id!r}")
        if not isinstance(model_blocks, dict):
            raise CatalogError(f"{BLOCKS_FILE}: {model_id} must map blocks to values")
        for key, value in model_blocks.items():
            if key not in blocks:
                raise CatalogError(
                    f"{BLOCKS_FILE}: {model_id}'s {key} block is one no provider takes"
                )
            blocks[key][model_id] = value
    return blocks
