"""Model files: resolving one, alone or in a catalog, and their schema.

A model file names a loader, a runtime and steps, and says what goes
in and out:

```yaml
loader: {use: mlflow, uri: "mlflow-artifacts:/…/model.onnx"}
runtime: {use: onnx, tensorrt: true, threads: 4}
input: {tile_size: 512, mpp: 0.468}
output: {kind: scores, channels: 1, size: 1, range: [0, 1]}
batching: {max_size: 256, wait_s: 0.05}
resources: {gpus: 1, cpus: 4, memory_gib: 12}
```
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from pydantic import Field, create_model

from components import Loader, Runtime, Step, slots
from model_service.catalog.context import ResolveContext
from model_service.catalog.errors import CatalogInvalid
from model_service.catalog.models.declaration import Model
from model_service.catalog.models.fields import Fields
from model_service.catalog.schema import usable
from model_service.registry import default_registry


if TYPE_CHECKING:
    from pydantic import BaseModel

    from model_service.registry import Registry


FOLDER = "models"
SUMMARY = "Model files: a loader, a runtime, steps, and what goes in and out."


def build_model(document: dict[str, Any], ctx: ResolveContext) -> Model | None:
    """Build the components and validate the blocks of one model file.

    Args:
        document: The model file after the environment's patch.
        ctx: Resolves components and collects errors.

    Returns:
        The `Model`, or None if anything failed.
    """
    fields = ctx.fields(Fields, document)
    loader = ctx.component(slots.LOADER, document.get("loader"), ("loader",))
    runtime = ctx.component(slots.RUNTIME, document.get("runtime"), ("runtime",))
    pre = cast("list[Step]", ctx.components(slots.PRE, document.get("pre"), ("pre",)))
    post = cast(
        "list[Step]", ctx.components(slots.POST, document.get("post"), ("post",))
    )
    if fields is None or loader is None or runtime is None or ctx.errors:
        return None
    return Model(
        id=ctx.document_id,
        loader=cast("Loader", loader),
        runtime=cast("Runtime", runtime),
        pre=tuple(pre),
        post=tuple(post),
        input=fields.input,
        output=fields.output,
        batching=fields.batching,
        resources=fields.resources,
        serving=fields.serving,
        declaration=document,
    )


def model_schema(registry: Registry) -> type[BaseModel]:
    """A model file, with the loaders, runtimes and steps there are.

    Args:
        registry: The components files may name.

    Returns:
        The file's model, for its JSON Schema.
    """
    pre = list[usable(registry, slots.PRE)]  # type: ignore[misc, valid-type]
    post = list[usable(registry, slots.POST)]  # type: ignore[misc, valid-type]
    return create_model(
        "model",
        __base__=Fields,
        __doc__=SUMMARY,
        loader=(usable(registry, slots.LOADER), _field("loader")),
        runtime=(usable(registry, slots.RUNTIME), _field("runtime")),
        pre=(pre, _field("pre")),
        post=(post, _field("post")),
    )


def _field(name: str) -> Any:
    """A component field of the file schema, with `Fields`' description and default.

    Args:
        name: The field, e.g. `loader`.

    Returns:
        The pydantic field.
    """
    info = Fields.model_fields[name]
    if info.is_required():
        return Field(description=info.description)
    return Field(info.default, description=info.description)


def resolve_model(
    model_id: str, document: dict[str, Any], registry: Registry | None = None
) -> Model:
    """Resolve one model document, as the host does from its user_config.

    The document must be the one the catalog resolved (after its environment's
    patch), so the host and the catalog agree on the digest.

    Args:
        model_id: The model ID.
        document: The model document.
        registry: The components it may name; this repo's by default.

    Returns:
        The model.

    Raises:
        CatalogInvalid: With every error, attributed to
            `models/<id>.yaml`.
    """
    ctx = ResolveContext(
        registry or default_registry(),
        lambda _: f"{FOLDER}/{model_id}.yaml",
        model_id,
    )
    model = build_model(document, ctx)
    if ctx.errors or model is None:
        raise CatalogInvalid(ctx.errors)
    return model
