"""Model catalog derived from the deployed model YAML files.

Helm renders every `deploy/helm/rayservice/models/<id>.yaml` into one JSON list. Each
entry has an `inference` declaration and the Ray Serve application (`serve`)
that Helm also injects into the RayService, so the catalog and the running
models come from the same files.

Contract parameters the model already reads from its `user_config` are taken
from there; `inference` declares only what the model does not read itself. Any
other top-level block belongs to a provider (e.g. how that provider offers
the model): the catalog keeps it for that provider, unread, and rejects a
block no provider takes.
"""

from __future__ import annotations

import json
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


class InferenceDeclaration(BaseModel):
    # Besides the contract, the fields its model class takes from here.
    model_config = ConfigDict(frozen=True, extra="allow")

    contract: InferenceContract


class _ServeDeployment(BaseModel):
    model_config = ConfigDict(extra="allow")

    user_config: dict[str, object] = Field(default_factory=dict)


class _ServeApplication(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")]
    route_prefix: str
    deployments: Annotated[list[_ServeDeployment], Field(min_length=1)]


class ModelDocument(BaseModel):
    # Other blocks are providers'.
    model_config = ConfigDict(extra="allow")

    inference: InferenceDeclaration
    serve: _ServeApplication


class CatalogError(ValueError):
    pass


def model_binding(document: ModelDocument) -> ModelBinding:
    """Build a model binding from one model document.

    Each field of the contract's model class comes from the `inference` block if
    the class lists it in `inference_fields`, otherwise from the first
    deployment's `user_config`.

    Args:
        document: A parsed model YAML document.

    Returns:
        The validated binding.

    Raises:
        CatalogError: If the route prefix is not `/<model id>`, a parameter is
            declared in the wrong place or missing, or the values are invalid.
    """
    model_id = document.serve.name
    if document.serve.route_prefix != f"/{model_id}":
        raise CatalogError(f"model {model_id!r} must use route_prefix /{model_id}")
    inference = document.inference
    model_class = MODEL_CLASSES[inference.contract]
    from_inference = set(model_class.inference_fields)
    from_user_config = (
        model_class.model_fields.keys()
        - from_inference
        - {
            "id",
            "contract",
        }
    )
    declared = inference.model_extra or {}
    if extra := declared.keys() - from_inference:
        raise CatalogError(
            f"model {model_id!r} declares {', '.join(sorted(extra))} in inference; "
            f"{inference.contract} reads it from user_config"
        )
    user_config = document.serve.deployments[0].user_config
    values: dict[str, object] = {"id": model_id, "contract": inference.contract}
    for name, source, where in (
        *((n, user_config, "user_config") for n in sorted(from_user_config)),
        *((n, declared, "inference") for n in sorted(from_inference)),
    ):
        if name not in source:
            raise CatalogError(f"model {model_id!r} {where} lacks {name}")
        values[name] = source[name]
    try:
        return MODEL_BINDING.validate_python(values)
    except ValidationError as error:
        raise CatalogError(
            f"model {model_id!r} declaration is invalid: {error}"
        ) from error


class ModelCatalog:
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

        Args:
            documents: The decoded JSON, expected to be a list of model documents.
            provider_blocks: The top-level keys providers read.

        Returns:
            The catalog.

        Raises:
            CatalogError: If `documents` is not a list, a document is invalid,
                a document has a block no provider takes, or a binding is invalid.
        """
        if not isinstance(documents, list):
            raise CatalogError("model catalog must be a list")
        parsed = []
        for index, item in enumerate(documents):
            try:
                parsed.append(ModelDocument.model_validate(item))
            except ValidationError as error:
                raise CatalogError(
                    f"model file {_document_name(item, index)} is invalid: {error}"
                ) from error
        blocks: dict[str, dict[str, object]] = {key: {} for key in provider_blocks}
        for document in parsed:
            for key, value in (document.model_extra or {}).items():
                if key not in blocks:
                    raise CatalogError(
                        f"model {document.serve.name!r} has a {key!r} block "
                        "no provider takes"
                    )
                blocks[key][document.serve.name] = value
        return cls([model_binding(document) for document in parsed], blocks)

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
            CatalogError: If the file cannot be read or decoded, or its content is invalid.
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
