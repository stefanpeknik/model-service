"""`/v1/builders`: the builder catalog."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import APIRouter

# FastAPI reads the view annotations at runtime.
from model_service.builders.contracts import (  # noqa: TC001
    InferenceContract,
    JsonValue,
)
from model_service.builders.registry import BUILDERS


@dataclass(frozen=True, slots=True)
class BuilderInputView:
    required: bool
    media_type: str | None


@dataclass(frozen=True, slots=True)
class BuilderOutputView:
    kind: str
    media_type: str | None


@dataclass(frozen=True, slots=True)
class BuilderView:
    id: str
    accepts: InferenceContract
    inputs: dict[str, BuilderInputView]
    outputs: dict[str, BuilderOutputView]
    configuration: dict[str, JsonValue]


def create_router() -> APIRouter:
    """Build the `/v1/builders` router: the builder catalog.

    Returns:
        The router.
    """
    router = APIRouter(prefix="/v1/builders")

    @router.get("")
    async def list_builders() -> list[BuilderView]:
        return [
            BuilderView(
                id=spec.id,
                accepts=spec.accepts,
                inputs={
                    name: BuilderInputView(item.required, item.media_type)
                    for name, item in spec.inputs.items()
                },
                outputs={
                    name: BuilderOutputView(item.kind, item.media_type)
                    for name, item in spec.outputs.items()
                },
                configuration=spec.configuration().model_dump(mode="json"),
            )
            for spec in BUILDERS.values()
        ]

    return router
