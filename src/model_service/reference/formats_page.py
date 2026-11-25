"""`docs/reference/formats.md`: every data format, from its module docstring."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from components import Publisher, Sink
from model_service.formats.known import FORMATS
from model_service.reference.markdown import GENERATED, code, docstring, facts


if TYPE_CHECKING:
    from model_service.registry import Registry


def formats_page(registry: Registry) -> str:
    """Render the data-formats reference.

    Args:
        registry: The components, to list who writes and reads each format.

    Returns:
        The page.
    """
    parts = [
        GENERATED,
        "# Data formats",
        "A data format is a named, versioned file shape. Sinks write one, "
        "publishers read one, and callers open them with the same module under "
        "`src/model_service/formats/` or the RationAI SDK's `rationai.formats`. "
        "A stored output carries its media type; a reader refuses a version it "
        "doesn't know. `regions` goes the other way: callers send it as an input.",
    ]
    for name, found in sorted(FORMATS.items()):
        module = importlib.import_module(found.module)
        writers = [
            f"[`{c.name}`](components.md#{c.name}-{c.kind})"
            for c in registry
            if issubclass(c, Sink) and c.format == name
        ]
        readers = [
            f"[`{c.name}`](components.md#{c.name}-{c.kind}) as "
            + ", ".join(map(code, sorted(c.emits)))
            for c in registry
            if issubclass(c, Publisher) and c.reads_format == name
        ]
        path = found.module.replace(".", "/")
        rows = [
            ["Media type", code(found.media_type)],
            ["File suffix", code(f".{found.extension}")],
            ["Written by", ", ".join(writers) or "no sink yet"],
            ["Published to EMPAIA by", "; ".join(readers) or "no publisher yet"],
            ["Code", f"[`src/{path}.py`](../../src/{path}.py)"],
        ]
        parts += [f"## `{name}`", facts(rows), docstring(module)]
    return "\n\n".join(parts) + "\n"
