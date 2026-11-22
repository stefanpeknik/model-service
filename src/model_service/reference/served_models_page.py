"""`docs/reference/served-models.md`: every served-model file of the catalog."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from model_service.catalog.analyses import FOLDER as ANALYSES
from model_service.catalog.served_models import FOLDER, ScoresOutput
from model_service.catalog.served_models.fields import Fields
from model_service.reference.markdown import GENERATED, code, facts, table


if TYPE_CHECKING:
    from model_service.catalog import Catalog
    from model_service.catalog.served_models import ServedModel


PAGE = Path("docs/reference/served-models.md")


def served_models_page(catalog: Catalog, root: Path) -> str:
    """Render the served-models reference.

    Args:
        catalog: The base catalog.
        root: The catalog directory, for environment patches.

    Returns:
        The page.
    """
    callers: dict[str, list[str]] = {}
    for entry in catalog.entries(ANALYSES).values():
        for model_id in entry.value.models:
            callers.setdefault(model_id, []).append(
                f"[`{entry.id}`](analyses.md#{entry.id})"
            )
    parts = [
        GENERATED,
        "# Served models",
        "Each file in `catalog/served-models/` is one trained model behind the "
        "generic host. The file name is the model ID and its raw route, "
        "`POST /<id>/` on Ray Serve; `GET /v1/models/<id>` on the API returns "
        "its declaration and digest. To add one, see "
        "[Add a model](../guides/add-a-model.md); the components named below "
        "are in [Components](components.md).",
        "## Keys of a served-model file",
        table(
            ["Key", "Description"],
            [
                [code(name), info.description or ""]
                for name, info in Fields.model_fields.items()
            ],
        ),
    ]
    for model_id, entry in catalog.entries(FOLDER).items():
        model: ServedModel = entry.value
        written = entry.document
        rows = [
            ["Weights", code(written["weights"])],
            ["Runtime", code(written["runtime"])],
            ["Pre", ", ".join(map(code, written.get("pre") or [])) or "none"],
            ["Post", ", ".join(map(code, written.get("post") or [])) or "none"],
            ["Input", f"{model.input.tile_size} px tiles at {model.input.mpp} µm/px"],
            ["Output", _output(model)],
            ["Raw reply", _reply(model)],
            [
                "Batching",
                f"up to {model.batching.max_size} tiles, waiting "
                f"{model.batching.wait_s} s to fill",
            ],
            ["Resources", code(written["resources"])],
            ["Called by", ", ".join(callers.get(model_id, [])) or "no analysis"],
            ["Patched in", _patches(root, model_id) or "no environment"],
            ["File", f"[`catalog/{FOLDER}/{model_id}.yaml`](../../{entry.file})"],
        ]
        parts += [f"## `{model_id}`", facts(rows)]
    return "\n\n".join(parts) + "\n"


def _output(model: ServedModel) -> str:
    output = model.output
    if isinstance(output, ScoresOutput):
        shape = f"{output.channels}, {output.size}, {output.size}"
        limits = (
            f", in [{output.range[0]:g}, {output.range[1]:g}]" if output.range else ""
        )
        return f"scores, {output.dtype} ({shape}) per tile{limits}"
    return f"embedding, {output.dtype}: {', '.join(output.summaries)}"


def _reply(model: ServedModel) -> str:
    output = model.output
    if isinstance(output, ScoresOutput):
        if output.channels == 1 and output.size == 1:
            return "the score as a JSON float"
        return f"LZ4-compressed {output.dtype} bytes"
    return "LZ4-compressed features with an `x-output-shape` header"


def _patches(root: Path, model_id: str) -> str:
    """The environments that patch the file, with the keys each changes."""
    found = []
    for path in sorted(root.glob(f"environments/*/{FOLDER}/{model_id}.yaml")):
        keys = ", ".join(code(k) for k in yaml.safe_load(path.read_text()) or {})
        found.append(f"{path.parts[-3]} ({keys})")
    return "; ".join(found)
