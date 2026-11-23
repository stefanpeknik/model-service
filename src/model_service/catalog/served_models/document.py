"""Served-model files: resolving one, alone or in a catalog, and their schema.

A served-model file names a loader, a runtime and steps, and says what goes
in and out:

```yaml
weights: {use: mlflow, uri: "mlflow-artifacts:/…/model.onnx"}
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

from components.base import Loader, Runtime, Step, slots
from model_service.catalog.context import ResolveContext
from model_service.catalog.errors import CatalogInvalid
from model_service.catalog.schema import usable
from model_service.catalog.served_models.declaration import ServedModel
from model_service.catalog.served_models.fields import Fields
from model_service.registry import default_registry


if TYPE_CHECKING:
    from pydantic import BaseModel

    from components.base import Slot
    from model_service.catalog.errors import YamlPath
    from model_service.registry import Registry


FOLDER = "served-models"
SUMMARY = "Served-model files: a loader, a runtime, steps, and what goes in and out."


def build_served_model(
    document: dict[str, Any], ctx: ResolveContext
) -> ServedModel | None:
    """Build the components and validate the blocks of one served-model file.

    Args:
        document: The served-model file after the environment's patch.
        ctx: Resolves components and collects errors.

    Returns:
        The `ServedModel`, or None if anything failed.
    """
    fields = ctx.fields(Fields, document)
    weights = ctx.component(slots.WEIGHTS, document.get("weights"), ("weights",))
    runtime = ctx.component(slots.RUNTIME, document.get("runtime"), ("runtime",))
    pre = _steps(ctx, slots.PRE, document.get("pre"), ("pre",))
    post = _steps(ctx, slots.POST, document.get("post"), ("post",))
    if fields is None or weights is None or runtime is None or ctx.errors:
        return None
    return ServedModel(
        id=ctx.document_id,
        weights=cast("Loader", weights),
        runtime=cast("Runtime", runtime),
        pre=pre,
        post=post,
        input=fields.input,
        output=fields.output,
        batching=fields.batching,
        resources=fields.resources,
        serving=fields.serving,
        declaration=document,
    )


def served_model_schema(registry: Registry) -> type[BaseModel]:
    """A served-model file, with the loaders, runtimes and steps there are.

    Args:
        registry: The components files may name.

    Returns:
        The file's model, for its JSON Schema.
    """
    pre = list[usable(registry, slots.PRE)]  # type: ignore[misc, valid-type]
    post = list[usable(registry, slots.POST)]  # type: ignore[misc, valid-type]
    return create_model(
        "served model",
        __base__=Fields,
        __doc__=SUMMARY,
        weights=(usable(registry, slots.WEIGHTS), _field("weights")),
        runtime=(usable(registry, slots.RUNTIME), _field("runtime")),
        pre=(pre, _field("pre")),
        post=(post, _field("post")),
    )


def _field(name: str) -> Any:
    """A component field of the file schema, with `Fields`' description and default.

    Args:
        name: The field, e.g. `weights`.

    Returns:
        The pydantic field.
    """
    info = Fields.model_fields[name]
    if info.is_required():
        return Field(description=info.description)
    return Field(info.default, description=info.description)


def _steps(
    ctx: ResolveContext, slot: Slot, value: object, path: YamlPath
) -> tuple[Step, ...]:
    """Resolve `pre:` or `post:`; a step that fails is left out after its error.

    Args:
        ctx: Collects errors.
        slot: The slot.
        value: Its value.
        path: Where it is in the file.

    Returns:
        The steps that resolved, in order.
    """
    if value is None:
        return ()
    if not isinstance(value, list):
        ctx.error(
            path,
            "not a list",
            f"`{slot.key}: [{{use: <step>, …}}, …]`",
            f"write `{slot.key}:` as a list of steps",
        )
        return ()
    steps = [ctx.component(slot, item, (*path, i)) for i, item in enumerate(value)]
    return tuple(cast("Step", step) for step in steps if step is not None)


def resolve_served_model(
    model_id: str, document: dict[str, Any], registry: Registry | None = None
) -> ServedModel:
    """Resolve one served-model document, as the host does from its user_config.

    The document must be the one the catalog resolved (after its environment's
    patch), so the host and the catalog agree on the digest.

    Args:
        model_id: The model ID.
        document: The served-model document.
        registry: The components it may name; this repo's by default.

    Returns:
        The served model.

    Raises:
        CatalogInvalid: With every error, attributed to
            `served-models/<id>.yaml`.
    """
    ctx = ResolveContext(
        registry or default_registry(),
        lambda _: f"{FOLDER}/{model_id}.yaml",
        model_id,
    )
    model = build_served_model(document, ctx)
    if ctx.errors or model is None:
        raise CatalogInvalid(ctx.errors)
    return model
