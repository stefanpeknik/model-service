"""A resolved served model and the digest that identifies what it computes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from model_service.catalog.served_models.fields import (
        Batching,
        Input,
        Output,
        Resources,
        Serving,
    )
    from model_service.components import Component, Loader, Runtime, Step


@dataclass(frozen=True)
class ServedModel:
    """One served-model file, resolved: components built, blocks validated.

    Attributes:
        id: The model ID, the file name; also the Serve application's name.
        weights: The loader.
        runtime: The runtime.
        pre: Steps run on each tile before the runtime.
        post: Steps run on each output after the runtime.
        input: What the model takes.
        output: What the model returns.
        batching: How requests are batched.
        resources: What a replica reserves.
        serving: Scaling and queue limits.
    """

    id: str
    weights: Loader
    runtime: Runtime
    pre: tuple[Step, ...]
    post: tuple[Step, ...]
    input: Input
    output: Output
    batching: Batching
    resources: Resources
    serving: Serving

    @cached_property
    def digest(self) -> str:
        """A SHA-256 of everything that decides what the model computes.

        It covers the weights (whose options pin the revision), the runtime,
        `pre`, `post`, `input`, `output` and `batching.max_size`, which shapes
        a TensorRT profile. `resources`, `serving` and `batching.wait_s` only
        change how the model is scheduled, so they leave it alone. Defaults
        count as written, so spelling one out keeps the digest.

        The analysis runner compares it with the running model's
        `describe()`, and the runtime cache directory is named after it.

        Returns:
            The hex digest.
        """
        canonical = {
            "weights": _component(self.weights),
            "runtime": _component(self.runtime),
            "pre": [_component(step) for step in self.pre],
            "post": [_component(step) for step in self.post],
            "input": self.input.model_dump(mode="json"),
            "output": self.output.model_dump(mode="json"),
            "batching": {"max_size": self.batching.max_size},
        }
        text = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode()).hexdigest()


def _component(component: Component) -> dict[str, Any]:
    return {"use": component.name, **component.options.model_dump(mode="json")}
