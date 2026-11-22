"""Model catalog derived from the deployed served-model and model files.

Served models (`catalog/served-models/<id>.yaml`) give their binding through
`served_model_binding`; `api/catalog_files.py` reads them from the mounted
catalog. Models not yet migrated still come from their old model file: Helm
renders every listed `deploy/helm/rayservice/models/<id>.yaml` into one JSON
list, adding each file's name as its `id`. The same files become the
RayService's applications, so the catalog and the running models cannot
drift apart.

Contract parameters come from the deployment's `user_config`, which the model
implementation reads too; `contract` names the contract. Any other top-level
block belongs to a provider (e.g. how that provider offers the model): the
catalog keeps it for that provider, unread.

A model file that is invalid, has a block no provider takes, or repeats an
earlier model's ID is logged and left out, so one bad file does not take the
other models down; only a catalog that cannot be read stops the API.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Annotated, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from model_service.catalog.catalog import Catalog
from model_service.catalog.served_models import ScoresOutput, legacy_config
from model_service.contracts.failures import UnknownModelError
from model_service.contracts.model_declarations import (
    MODEL_BINDING,
    MODEL_CLASSES,
    InferenceContract,
)


if TYPE_CHECKING:
    from collections.abc import Collection, Iterator, Mapping
    from pathlib import Path

    from model_service.catalog.served_models import ServedModel
    from model_service.contracts.model_declarations import ModelBinding


logger = logging.getLogger(__name__)


class _Deployment(BaseModel):
    """The part of a model's Ray Serve deployment the catalog reads.

    Attributes:
        user_config: The deployment's user_config.
    """

    model_config = ConfigDict(extra="allow")

    user_config: dict[str, object] = Field(default_factory=dict)


class ModelDocument(BaseModel):
    """One model file, with the `id` Helm adds; other blocks belong to providers.

    Attributes:
        id: The model ID, the file's name.
        contract: The model's inference contract.
        implementation: The code that serves it, `module.Class` in
            `inference/implementations/`; Helm builds the Ray application from it.
        deployment: The model's Ray Serve deployment options.
        application: Ray Serve options for the whole application, if any.
    """

    # Other blocks are providers'.
    model_config = ConfigDict(extra="allow")

    id: Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")]
    contract: InferenceContract
    implementation: Annotated[
        str, Field(pattern=r"^[a-z_][a-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*$")
    ]
    deployment: _Deployment
    application: dict[str, object] = Field(default_factory=dict)


class CatalogError(ValueError):
    """The model files do not form a valid catalog."""


def model_binding(document: ModelDocument) -> ModelBinding:
    """Build a model binding from one model document.

    Every field of the contract's model class comes from the deployment's
    `user_config`.

    Args:
        document: A parsed model YAML document.

    Returns:
        The validated binding.

    Raises:
        CatalogError: If a parameter is missing or the values are invalid.
    """
    model_class = MODEL_CLASSES[document.contract]
    user_config = document.deployment.user_config
    values: dict[str, object] = {"id": document.id, "contract": document.contract}
    for name in sorted(model_class.model_fields.keys() - {"id", "contract"}):
        if name not in user_config:
            raise CatalogError(f"model {document.id!r} user_config lacks {name}")
        values[name] = user_config[name]
    try:
        return MODEL_BINDING.validate_python(values)
    except ValidationError as error:
        raise CatalogError(
            f"model {document.id!r} declaration is invalid: {error}"
        ) from error


def served_model_binding(model: ServedModel) -> ModelBinding:
    """Derive today's model binding from a served model (transitional).

    A scores output binds as `spatial-scores`, an embedding as
    `tile-embedding`, with the fields the host's `get_config()` reports.

    Args:
        model: The resolved served model.

    Returns:
        The binding builders verify the hosted model against.
    """
    contract = (
        InferenceContract.SPATIAL_SCORES
        if isinstance(model.output, ScoresOutput)
        else InferenceContract.TILE_EMBEDDING
    )
    return MODEL_BINDING.validate_python(
        {"id": model.id, "contract": contract, **legacy_config(model)}
    )


class ModelCatalogView(Protocol):
    """What the API's parts read from the catalog, whether fixed or reloaded."""

    @property
    def resolved(self) -> Catalog:
        """The served models and analyses, as the catalog loader resolved them."""
        ...

    def get(self, model_id: str) -> ModelBinding:
        """Look up a deployed model.

        Args:
            model_id: The model ID.

        Returns:
            The model binding.

        Raises:
            UnknownModelError: If no model has this ID.
        """
        ...

    def provider_block(self, key: str) -> dict[str, object]:
        """Return each model's `key` block, by model ID.

        Args:
            key: The provider's top-level block key.

        Returns:
            Model IDs mapped to their block.
        """
        ...

    def __iter__(self) -> Iterator[ModelBinding]:
        """Iterate over the model bindings.

        Returns:
            An iterator over the bindings in catalog order.
        """
        ...

    def __len__(self) -> int:
        """Count the models.

        Returns:
            The number of models.
        """
        ...


class ModelCatalog:
    """The deployed models by id, with each provider's blocks from the model files."""

    def __init__(
        self,
        models: list[ModelBinding],
        blocks: Mapping[str, Mapping[str, object]] | None = None,
        *,
        resolved: Catalog | None = None,
    ) -> None:
        """Index models by ID and keep the provider blocks.

        Args:
            models: Model bindings.
            blocks: Provider blocks by block key, then by model ID.
            resolved: The served models and analyses the catalog loader
                resolved; None for none.

        Raises:
            CatalogError: If two models share an ID.
        """
        self._models = {model.id: model for model in models}
        if len(self._models) != len(models):
            raise CatalogError("model IDs must be unique")
        self._blocks = {key: dict(value) for key, value in (blocks or {}).items()}
        self.resolved = resolved or Catalog(None)

    @classmethod
    def from_documents(
        cls, documents: object, provider_blocks: Collection[str] = ()
    ) -> ModelCatalog:
        """Build the catalog from the JSON list Helm renders from the model YAML files.

        A document that is invalid, has a block no provider takes, or repeats an
        earlier document's model ID is logged and left out.

        Args:
            documents: The decoded JSON, expected to be a list of model documents.
            provider_blocks: The top-level keys providers read.

        Returns:
            The catalog of the valid documents.

        Raises:
            CatalogError: If `documents` is not a list.
        """
        if not isinstance(documents, list):
            raise CatalogError("model catalog must be a list")
        models: list[ModelBinding] = []
        blocks: dict[str, dict[str, object]] = {key: {} for key in provider_blocks}
        for index, item in enumerate(documents):
            name = _document_name(item, index)
            try:
                document = ModelDocument.model_validate(item)
                extra = document.model_extra or {}
                if unknown := sorted(extra.keys() - blocks.keys()):
                    raise CatalogError(
                        f"its {', '.join(unknown)} block is one no provider takes"
                    )
                model = model_binding(document)
            except (ValidationError, CatalogError) as error:
                logger.error("model file %s is left out: %s", name, error)
                continue
            if any(known.id == model.id for known in models):
                logger.error(
                    "model file %s is left out: an earlier file has its ID", name
                )
                continue
            models.append(model)
            for key, value in extra.items():
                blocks[key][model.id] = value
        return cls(models, blocks)

    @classmethod
    def from_file(
        cls, path: Path, provider_blocks: Collection[str] = ()
    ) -> ModelCatalog:
        """Build the catalog from the rendered JSON file.

        Args:
            path: The catalog JSON file.
            provider_blocks: The top-level keys providers read.

        Returns:
            The catalog.

        Raises:
            CatalogError: If the file cannot be read or decoded, or is not a list.
        """
        try:
            documents = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise CatalogError(f"cannot read model catalog {path}") from error
        return cls.from_documents(documents, provider_blocks)

    def provider_block(self, key: str) -> dict[str, object]:
        """Return each model's `key` block, by model ID.

        Args:
            key: The provider's top-level block key.

        Returns:
            A copy mapping model IDs to their block; models without one are absent.
        """
        return dict(self._blocks.get(key, {}))

    def get(self, model_id: str) -> ModelBinding:
        """Look up a deployed model.

        Args:
            model_id: The model ID.

        Returns:
            The model binding.

        Raises:
            UnknownModelError: If no model has this ID.
        """
        try:
            return self._models[model_id]
        except KeyError as error:
            raise UnknownModelError(model_id) from error

    def __iter__(self) -> Iterator[ModelBinding]:
        """Iterate over the model bindings.

        Returns:
            An iterator over the bindings in catalog order.
        """
        return iter(self._models.values())

    def __len__(self) -> int:
        """Count the models.

        Returns:
            The number of models in the catalog.
        """
        return len(self._models)


def _document_name(item: object, index: int) -> str:
    """Name a model document for an error: its `id` if readable, else its position.

    Args:
        item: The raw document.
        index: Its position in the catalog list.

    Returns:
        A quoted model name, or `#<index>`.
    """
    name = item.get("id") if isinstance(item, dict) else None
    return repr(name) if isinstance(name, str) else f"#{index}"
