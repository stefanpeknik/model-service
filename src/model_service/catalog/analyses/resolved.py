"""An analysis resolved for one run: the document the control plane admits and the runner executes.

At admission the control plane resolves the analysis with the caller's settings into one
JSON document. Every component appears with all of its options (defaults,
filled-in geometry and the caller's overrides included), each served model it
calls with its declaration and digest, and a digest over all of it:

```json
{
  "id": "prostate-heatmap",
  "summary": "...",
  "source": {"use": "slide_tiles", "tile_size": 512, "stride_fraction": 0.5, ...},
  "steps": [{"model": "prostate-classifier-1"}],
  "sinks": [{"use": "score_grid", "output": "heatmap"}],
  "served_models": {"prostate-classifier-1": {"digest": "...", "declaration": {...}}},
  "digest": "..."
}
```

The runner reads it back with `load_resolved`, so it runs exactly what the
control plane validated even if the catalog changed in between, and compares each
served model's `describe()` digest with the one recorded here. The document
doubles as the run's provenance record.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from pydantic import ValidationError

from model_service.catalog.analyses.document import FOLDER, build_analysis
from model_service.catalog.analyses.model_step import ModelStep
from model_service.catalog.context import ResolveContext
from model_service.catalog.errors import CatalogInvalid
from model_service.catalog.served_models import resolve_served_model
from model_service.contracts.failures import InvalidInvocation, UnknownAnalysisError
from model_service.registry import default_registry


if TYPE_CHECKING:
    from collections.abc import Mapping

    from model_service.catalog.analyses.declaration import Analysis
    from model_service.catalog.catalog import Catalog
    from model_service.catalog.served_models import ServedModel
    from model_service.registry import Registry


DIGEST = "digest"


class ResolvedAnalysisInvalid(ValueError):
    """A resolved document that was changed after admission or doesn't resolve."""


@dataclass(frozen=True)
class ResolvedAnalysis:
    """An analysis with one run's settings, and its document.

    Attributes:
        analysis: The analysis, settings applied, ready for `run_analysis`.
        document: The JSON document, digest included.
    """

    analysis: Analysis
    document: dict[str, Any]

    @property
    def digest(self) -> str:
        """The digest over the whole document."""
        return cast("str", self.document[DIGEST])

    @property
    def model_digests(self) -> dict[str, str]:
        """Each served model's digest as the control plane resolved it, by ID."""
        return {
            model_id: served[DIGEST]
            for model_id, served in self.document["served_models"].items()
        }


def resolve_run(
    catalog: Catalog, analysis_id: str, settings: Mapping[str, Any] | None = None
) -> ResolvedAnalysis:
    """Resolve an analysis of the catalog with a caller's per-run settings.

    Args:
        catalog: The deployed catalog.
        analysis_id: The analysis to run.
        settings: Per-run settings by flat name; None keeps the file's.

    Returns:
        The analysis and its document.

    Raises:
        UnknownAnalysisError: If the catalog has no such analysis.
        InvalidInvocation: For a setting the analysis doesn't take or an
            invalid value; the message names the settings.
    """
    analysis = catalog.analyses.get(analysis_id)
    if analysis is None:
        raise UnknownAnalysisError(analysis_id)
    try:
        analysis = analysis.with_settings(settings or {})
    except ValidationError as error:
        names = sorted({str(e["loc"][0]) for e in error.errors() if e["loc"]})
        raise InvalidInvocation(
            f"invalid settings of {analysis_id}: {', '.join(names)}"
        ) from error
    except ValueError as error:
        raise InvalidInvocation(str(error)) from error
    document: dict[str, Any] = {
        "id": analysis.id,
        "summary": analysis.summary,
        "source": analysis.source.spelled_out(),
        "steps": [
            {"model": s.model.id} if isinstance(s, ModelStep) else s.spelled_out()
            for s in analysis.steps
        ],
        "sinks": [s.spelled_out() for s in analysis.sinks],
        "served_models": {
            model_id: {DIGEST: model.digest, "declaration": model.declaration}
            for model_id, model in analysis.models.items()
        },
    }
    return ResolvedAnalysis(analysis, document | {DIGEST: _digest(document)})


def load_resolved(
    document: Mapping[str, Any], registry: Registry | None = None
) -> ResolvedAnalysis:
    """Read a resolved document back into the analysis it describes.

    Args:
        document: What `resolve_run` produced, e.g. from a run request.
        registry: The components it may name; this repo's by default.

    Returns:
        The analysis and the document.

    Raises:
        ResolvedAnalysisInvalid: If the digest doesn't match the content, a
            served model's declaration doesn't give its recorded digest, or
            the document doesn't resolve.
    """
    registry = registry or default_registry()
    document = dict(document)
    body = {k: v for k, v in document.items() if k != DIGEST}
    try:
        if document.get(DIGEST) != _digest(body):
            raise ResolvedAnalysisInvalid(
                "the analysis document does not match its digest"
            )
        analysis_id = body["id"]
        served: dict[str, ServedModel] = {}
        for model_id, item in body["served_models"].items():
            model = resolve_served_model(model_id, item["declaration"], registry)
            if model.digest != item[DIGEST]:
                raise ResolvedAnalysisInvalid(
                    f"served model {model_id!r} resolves to another digest here"
                )
            served[model_id] = model
        fields = ("summary", "source", "steps", "sinks")
        analysis = _resolve_analysis(
            analysis_id, {k: body[k] for k in fields}, served, registry
        )
    except (KeyError, TypeError, AttributeError) as error:
        raise ResolvedAnalysisInvalid("the analysis document is malformed") from error
    except CatalogInvalid as error:
        raise ResolvedAnalysisInvalid(
            f"the analysis document does not resolve: {error}"
        ) from error
    return ResolvedAnalysis(analysis, document)


def _resolve_analysis(
    analysis_id: str,
    document: dict[str, Any],
    served: dict[str, ServedModel],
    registry: Registry,
) -> Analysis:
    """Resolve an analysis document against the given served models alone.

    Args:
        analysis_id: The analysis ID.
        document: The source, steps and sinks with every option.
        served: The served models it calls, by ID.
        registry: The components it may name.

    Returns:
        The analysis.

    Raises:
        ResolvedAnalysisInvalid: If the document doesn't resolve.
    """
    ctx = ResolveContext(
        registry,
        lambda _: f"{FOLDER}/{analysis_id}.yaml",
        analysis_id,
        served_models=served,
    )
    analysis = build_analysis(document, ctx)
    if ctx.errors or analysis is None:
        raise CatalogInvalid(ctx.errors)
    return analysis


def _digest(body: Mapping[str, Any]) -> str:
    """SHA-256 of the canonical JSON of a document without its digest.

    Args:
        body: The document, digest left out.

    Returns:
        The hex digest.
    """
    text = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()
