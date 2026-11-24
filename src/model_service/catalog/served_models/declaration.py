"""A resolved served model and the digest that identifies what it computes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from components.base import Loader, Runtime, Step
    from model_service.catalog.served_models.fields import (
        Batching,
        Input,
        Output,
        Resources,
        Serving,
    )


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
        declaration: The file after the environment's patch, as the chart
            hands it to the host and an analysis run records it.
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
    declaration: dict[str, Any] = field(compare=False, repr=False)

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
            "weights": self.weights.spelled_out(),
            "runtime": self.runtime.spelled_out(),
            "pre": [step.spelled_out() for step in self.pre],
            "post": [step.spelled_out() for step in self.post],
            "input": self.input.model_dump(mode="json"),
            "output": self.output.model_dump(mode="json"),
            "batching": {"max_size": self.batching.max_size},
        }
        text = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode()).hexdigest()
