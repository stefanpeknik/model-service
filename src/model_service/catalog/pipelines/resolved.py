"""A pipeline resolved for one run: the document the control plane admits and the runner executes.

At admission the control plane resolves the pipeline with the caller's settings into one
JSON document. Every component appears with all of its options (defaults,
filled-in geometry and the caller's overrides included), each model it
calls with its declaration and digest, and a digest over all of it:

```json
{
  "id": "prostate-heatmap",
  "summary": "...",
  "source": {"use": "slide_tiles", "tile_size": 512, "stride_fraction": 0.5, ...},
  "steps": [{"model": "prostate-classifier-1"}],
  "sinks": [{"use": "score_grid", "output": "heatmap"}],
  "models": {"prostate-classifier-1": {"digest": "...", "declaration": {...}}},
  "digest": "..."
}
```

The runner reads it back with `load_resolved`, so it runs exactly what the
control plane validated even if the catalog changed in between, and compares each
model's `describe()` digest with the one recorded here. The document
doubles as the run's provenance record.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from pydantic import ValidationError

from model_service.catalog.context import ResolveContext
from model_service.catalog.errors import CatalogInvalid
from model_service.catalog.models import resolve_model
from model_service.catalog.pipelines.document import FOLDER, build_pipeline
from model_service.catalog.pipelines.model_step import ModelStep
from model_service.contracts.failures import InvalidInvocation, UnknownPipelineError
from model_service.registry import default_registry


if TYPE_CHECKING:
    from collections.abc import Mapping

    from model_service.catalog.catalog import Catalog
    from model_service.catalog.models import Model
    from model_service.catalog.pipelines.declaration import Pipeline
    from model_service.registry import Registry


DIGEST = "digest"


class ResolvedPipelineInvalid(ValueError):
    """A resolved document that was changed after admission or doesn't resolve."""


@dataclass(frozen=True)
class ResolvedPipeline:
    """A pipeline with one run's settings, and its document.

    Attributes:
        pipeline: The pipeline, settings applied, ready for `run_pipeline`.
        document: The JSON document, digest included.
    """

    pipeline: Pipeline
    document: dict[str, Any]

    @property
    def digest(self) -> str:
        """The digest over the whole document."""
        return cast("str", self.document[DIGEST])

    @property
    def model_digests(self) -> dict[str, str]:
        """Each model's digest as the control plane resolved it, by ID."""
        return {
            model_id: served[DIGEST]
            for model_id, served in self.document["models"].items()
        }


def resolve_run(
    catalog: Catalog, pipeline_id: str, settings: Mapping[str, Any] | None = None
) -> ResolvedPipeline:
    """Resolve a pipeline of the catalog with a caller's per-run settings.

    Args:
        catalog: The deployed catalog.
        pipeline_id: The pipeline to run.
        settings: Per-run settings by flat name; None keeps the file's.

    Returns:
        The pipeline and its document.

    Raises:
        UnknownPipelineError: If the catalog has no such pipeline.
        InvalidInvocation: For a setting the pipeline doesn't take or an
            invalid value; the message names the settings.
    """
    pipeline = catalog.pipelines.get(pipeline_id)
    if pipeline is None:
        raise UnknownPipelineError(pipeline_id)
    try:
        pipeline = pipeline.with_settings(settings or {})
    except ValidationError as error:
        names = sorted({str(e["loc"][0]) for e in error.errors() if e["loc"]})
        raise InvalidInvocation(
            f"invalid settings of {pipeline_id}: {', '.join(names)}"
        ) from error
    except ValueError as error:
        raise InvalidInvocation(str(error)) from error
    document: dict[str, Any] = {
        "id": pipeline.id,
        "summary": pipeline.summary,
        "source": pipeline.source.spelled_out(),
        "steps": [
            {"model": s.model.id} if isinstance(s, ModelStep) else s.spelled_out()
            for s in pipeline.steps
        ],
        "sinks": [s.spelled_out() for s in pipeline.sinks],
        "models": {
            model_id: {DIGEST: model.digest, "declaration": model.declaration}
            for model_id, model in pipeline.models.items()
        },
    }
    return ResolvedPipeline(pipeline, document | {DIGEST: _digest(document)})


def load_resolved(
    document: Mapping[str, Any], registry: Registry | None = None
) -> ResolvedPipeline:
    """Read a resolved document back into the pipeline it describes.

    Args:
        document: What `resolve_run` produced, e.g. from a run request.
        registry: The components it may name; this repo's by default.

    Returns:
        The pipeline and the document.

    Raises:
        ResolvedPipelineInvalid: If the digest doesn't match the content, a
            model's declaration doesn't give its recorded digest, or
            the document doesn't resolve.
    """
    registry = registry or default_registry()
    document = dict(document)
    body = {k: v for k, v in document.items() if k != DIGEST}
    try:
        if document.get(DIGEST) != _digest(body):
            raise ResolvedPipelineInvalid(
                "the pipeline document does not match its digest"
            )
        pipeline_id = body["id"]
        served: dict[str, Model] = {}
        for model_id, item in body["models"].items():
            model = resolve_model(model_id, item["declaration"], registry)
            if model.digest != item[DIGEST]:
                raise ResolvedPipelineInvalid(
                    f"model {model_id!r} resolves to another digest here"
                )
            served[model_id] = model
        fields = ("summary", "source", "steps", "sinks")
        pipeline = _resolve_pipeline(
            pipeline_id, {k: body[k] for k in fields}, served, registry
        )
    except (KeyError, TypeError, AttributeError) as error:
        raise ResolvedPipelineInvalid("the pipeline document is malformed") from error
    except CatalogInvalid as error:
        raise ResolvedPipelineInvalid(
            f"the pipeline document does not resolve: {error}"
        ) from error
    return ResolvedPipeline(pipeline, document)


def _resolve_pipeline(
    pipeline_id: str,
    document: dict[str, Any],
    served: dict[str, Model],
    registry: Registry,
) -> Pipeline:
    """Resolve a pipeline document against the given models alone.

    Args:
        pipeline_id: The pipeline ID.
        document: The source, steps and sinks with every option.
        served: The models it calls, by ID.
        registry: The components it may name.

    Returns:
        The pipeline.

    Raises:
        ResolvedPipelineInvalid: If the document doesn't resolve.
    """
    ctx = ResolveContext(
        registry,
        lambda _: f"{FOLDER}/{pipeline_id}.yaml",
        pipeline_id,
        models=served,
    )
    pipeline = build_pipeline(document, ctx)
    if ctx.errors or pipeline is None:
        raise CatalogInvalid(ctx.errors)
    return pipeline


def _digest(body: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of a document without its digest.

    Args:
        body: The document, digest left out.

    Returns:
        The hex digest.
    """
    text = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()
