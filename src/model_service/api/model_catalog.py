"""Model catalog derived from the deployed model YAML files.

Helm renders every `deploy/helm/rayservice/models/<id>.yaml` into one JSON list. Each
entry has an `inference` declaration and the Ray Serve application (`serve`)
that Helm also injects into the RayService, so the catalog and the running
models come from the same files.

Contract parameters come from the first deployment's `user_config`, which the
model implementation reads too; `inference` names only the contract. Any other
top-level block belongs to a provider (e.g. how that provider offers
the model): the catalog keeps it for that provider, unread.

A model file that is invalid, has a block no provider takes, or repeats an
earlier model's ID is logged and left out, so one bad file does not take the
other models down; only a catalog that cannot be read stops the API.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from model_service.contracts.failures import UnknownModelError
from model_service.contracts.model_declarations import (
    MODEL_BINDING,
    MODEL_CLASSES,
    InferenceContract,
)


if TYPE_CHECKING:
    from collections.abc import Collection, Iterator, Mapping
    from pathlib import Path

    from model_service.contracts.model_declarations import ModelBinding


logger = logging.getLogger(__name__)


class InferenceDeclaration(BaseModel):
    """A model file's `inference:` block.

    Attributes:
        contract: The model's inference contract; its parameters are in the
            first deployment's `user_config`.
    """

    # Other fields are allowed only to reject them with a pointer to user_config.
    model_config = ConfigDict(frozen=True, extra="allow")

    contract: InferenceContract


class _ServeDeployment(BaseModel):
    """The part of a Ray Serve deployment the catalog reads.

    Attributes:
        user_config: The deployment's user_config.
    """

    model_config = ConfigDict(extra="allow")

    user_config: dict[str, object] = Field(default_factory=dict)


class _ServeApplication(BaseModel):
    """The part of a model file's Ray Serve application the catalog reads.

    Attributes:
        name: The application name, which is the model id.
        route_prefix: The application's route prefix.
        deployments: Its deployments; the first one is the model.
    """

    model_config = ConfigDict(extra="allow")

    name: Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")]
    route_prefix: str
    deployments: Annotated[list[_ServeDeployment], Field(min_length=1)]


class ModelDocument(BaseModel):
    """One model file; blocks other than these belong to providers.

    Attributes:
        inference: The model's contract and parameters.
        serve: Its Ray Serve application.
    """

    # Other blocks are providers'.
    model_config = ConfigDict(extra="allow")

    inference: InferenceDeclaration
    serve: _ServeApplication


class CatalogError(ValueError):
    """The model files do not form a valid catalog."""


def model_binding(document: ModelDocument) -> ModelBinding:
    """Build a model binding from one model document.

    Every field of the contract's model class comes from the first deployment's
    `user_config`.

    Args:
        document: A parsed model YAML document.

    Returns:
        The validated binding.

    Raises:
        CatalogError: If the route prefix is not `/<model id>`, `inference` holds
            more than the contract, a parameter is missing, or the values are
            invalid.
    """
    model_id = document.serve.name
    if document.serve.route_prefix != f"/{model_id}":
        raise CatalogError(f"model {model_id!r} must use route_prefix /{model_id}")
    inference = document.inference
    if extra := sorted(inference.model_extra or {}):
        raise CatalogError(
            f"model {model_id!r} declares {', '.join(extra)} in inference; "
            "it belongs in user_config"
        )
    model_class = MODEL_CLASSES[inference.contract]
    user_config = document.serve.deployments[0].user_config
    values: dict[str, object] = {"id": model_id, "contract": inference.contract}
    for name in sorted(model_class.model_fields.keys() - {"id", "contract"}):
        if name not in user_config:
            raise CatalogError(f"model {model_id!r} user_config lacks {name}")
        values[name] = user_config[name]
    try:
        return MODEL_BINDING.validate_python(values)
    except ValidationError as error:
        raise CatalogError(
            f"model {model_id!r} declaration is invalid: {error}"
        ) from error


class ModelCatalog:
    """The deployed models by id, with each provider's blocks from the model files."""

    def __init__(
        self,
        models: list[ModelBinding],
        blocks: Mapping[str, Mapping[str, object]] | None = None,
    ) -> None:
        """Index models by ID and keep the provider blocks.

        Args:
            models: Model bindings.
            blocks: Provider blocks by block key, then by model ID.

        Raises:
            CatalogError: If two models share an ID.
        """
        self._models = {model.id: model for model in models}
        if len(self._models) != len(models):
            raise CatalogError("model IDs must be unique")
        self._blocks = {key: dict(value) for key, value in (blocks or {}).items()}

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
    """Name a model document for an error: its `serve.name` if readable, else its position.

    Args:
        item: The raw document.
        index: Its position in the catalog list.

    Returns:
        A quoted model name, or `#<index>`.
    """
    serve = item.get("serve") if isinstance(item, dict) else None
    name = serve.get("name") if isinstance(serve, dict) else None
    return repr(name) if isinstance(name, str) else f"#{index}"
