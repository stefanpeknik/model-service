"""Test-only document kinds that exercise the catalog loader's plug-in points.

`models/` stands in for served models (weights, runtime, post steps) and
`chains/` for analyses (a model reference, source, steps, sinks), so the loader
is proven before the real kinds exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field

from model_service.catalog import DocumentKind
from model_service.components import slots


if TYPE_CHECKING:
    from model_service.catalog import ResolveContext
    from model_service.components import Component


class _ModelFields(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tile_size: int = Field(gt=0, description="Tile edge the model takes.")


@dataclass
class TestModel:
    """A resolved `models/` document."""

    tile_size: int
    weights: Component | None
    runtime: Component | None
    post: list[Component | None]


@dataclass
class TestChain:
    """A resolved `chains/` document."""

    model: Any
    source: Component | None
    steps: list[Component | None]
    sinks: list[Component | None]


class Models(DocumentKind):
    folder = "models"
    title = "test model"

    def resolve(self, document: dict[str, Any], ctx: ResolveContext) -> TestModel:
        """Resolve the components and validate the remaining fields."""
        components = {"weights", "runtime", "post"}
        fields = ctx.fields(
            _ModelFields, {k: v for k, v in document.items() if k not in components}
        )
        return TestModel(
            tile_size=fields.tile_size if fields else 0,
            weights=ctx.component(slots.WEIGHTS, document.get("weights"), ("weights",)),
            runtime=ctx.component(slots.RUNTIME, document.get("runtime"), ("runtime",)),
            post=[
                ctx.component(slots.POST, value, ("post", i))
                for i, value in enumerate(document.get("post") or [])
            ],
        )


class Chains(DocumentKind):
    folder = "chains"
    title = "test chain"

    def resolve(self, document: dict[str, Any], ctx: ResolveContext) -> TestChain:
        """Resolve the model reference and every component."""
        return TestChain(
            model=ctx.reference("models", document.get("model"), ("model",)),
            source=ctx.component(slots.SOURCE, document.get("source"), ("source",)),
            steps=[
                ctx.component(slots.STEPS, value, ("steps", i))
                for i, value in enumerate(document.get("steps") or [])
            ],
            sinks=[
                ctx.component(slots.SINKS, value, ("sinks", i))
                for i, value in enumerate(document.get("sinks") or [])
            ],
        )


TEST_KINDS = (Models(), Chains())
