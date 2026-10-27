"""Model catalog derived from the deployed model YAML files.

Helm renders every `deploy/helm/rayservice/models/<id>.yaml` into one JSON list. Each
entry has an `inference` declaration and the Ray Serve application (`serve`)
that Helm also injects into the RayService, so the catalog and the running
models come from the same files.

Contract parameters the model already reads from its `user_config` are taken
from there; `inference` declares only what the model does not read itself.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from model_service.builders.contracts import (
    MODEL_BINDING,
    InferenceContract,
    UnknownModelError,
)
from model_service.builders.registry import BUILDERS
from model_service.providers.empaia.apps import EmpaiaListing  # noqa: TC001


if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping
    from pathlib import Path

    from model_service.builders.contracts import ModelBinding


class InferenceDeclaration(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    contract: InferenceContract
    mpp: Annotated[float, Field(gt=0, allow_inf_nan=False)] | None = None
    summaries: tuple[str, ...] | None = None


class _ServeDeployment(BaseModel):
    model_config = ConfigDict(extra="allow")

    user_config: dict[str, object] = Field(default_factory=dict)


class _ServeApplication(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: Annotated[str, Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")]
    route_prefix: str
    deployments: Annotated[list[_ServeDeployment], Field(min_length=1)]


class ModelDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inference: InferenceDeclaration
    serve: _ServeApplication
    # How the model is offered as EMPAIA apps; without it, it is not.
    empaia: EmpaiaListing | None = None


# Parameters each contract reads from user_config, and from the inference block.
_PARAMETERS: dict[InferenceContract, tuple[tuple[str, ...], tuple[str, ...]]] = {
    InferenceContract.SPATIAL_SCORES: (
        ("tile_size", "output_tile_size", "n_channels", "mpp"),
        (),
    ),
    InferenceContract.TILE_EMBEDDING: (("tile_size",), ("mpp", "summaries")),
}


class CatalogError(ValueError):
    pass


def model_binding(document: ModelDocument) -> ModelBinding:
    model_id = document.serve.name
    if document.serve.route_prefix != f"/{model_id}":
        raise CatalogError(f"model {model_id!r} must use route_prefix /{model_id}")
    inference = document.inference
    from_user_config, from_inference = _PARAMETERS[inference.contract]
    declared = {
        name for name in ("mpp", "summaries") if getattr(inference, name) is not None
    }
    if extra := declared - set(from_inference):
        raise CatalogError(
            f"model {model_id!r} declares {', '.join(sorted(extra))} in inference; "
            f"{inference.contract} reads it from user_config"
        )
    user_config = document.serve.deployments[0].user_config
    values: dict[str, object] = {"id": model_id, "contract": inference.contract}
    for name in from_user_config:
        if name not in user_config:
            raise CatalogError(f"model {model_id!r} user_config lacks {name}")
        values[name] = user_config[name]
    for name in from_inference:
        value = getattr(inference, name)
        if value is None:
            raise CatalogError(f"model {model_id!r} inference lacks {name}")
        values[name] = value
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
        empaia_listings: Mapping[str, EmpaiaListing] | None = None,
    ) -> None:
        self._models = {model.id: model for model in models}
        if len(self._models) != len(models):
            raise CatalogError("model IDs must be unique")
        self.empaia_listings = dict(empaia_listings or {})

    @classmethod
    def from_documents(cls, documents: object) -> ModelCatalog:
        if not isinstance(documents, list):
            raise CatalogError("model catalog must be a list")
        try:
            parsed = [ModelDocument.model_validate(item) for item in documents]
        except ValidationError as error:
            raise CatalogError(f"model catalog is invalid: {error}") from error
        return cls(
            [model_binding(document) for document in parsed],
            {
                document.serve.name: document.empaia
                for document in parsed
                if document.empaia is not None
            },
        )

    @classmethod
    def from_file(cls, path: Path) -> ModelCatalog:
        try:
            documents = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise CatalogError(f"cannot read model catalog {path}") from error
        return cls.from_documents(documents)

    def get(self, model_id: str) -> ModelBinding:
        try:
            return self._models[model_id]
        except KeyError as error:
            raise UnknownModelError(model_id) from error

    def __iter__(self) -> Iterator[ModelBinding]:
        return iter(self._models.values())

    def __len__(self) -> int:
        return len(self._models)

    @staticmethod
    def compatible_builders(model: ModelBinding) -> list[str]:
        return [spec.id for spec in BUILDERS.values() if spec.accepts == model.contract]
