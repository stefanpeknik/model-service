"""Every registered component passes its kind's conformance suite.

The repo's registry is checked as it grows; the test-only components prove the
suites pass on components that behave, and the broken ones below prove they
catch the mistakes they exist for.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pytest
from pydantic import Field

from model_service.components import (
    Col,
    Options,
    Registry,
    Sink,
    Step,
    default_registry,
)
from model_service.testing import conformance


if TYPE_CHECKING:
    from pathlib import Path

    from model_service.components import Batch, Component


COMPONENTS = [*default_registry(), *Registry.scan("tests.support.components")]


@pytest.mark.parametrize(
    "component", COMPONENTS, ids=[f"{c.__module__}:{c.name}" for c in COMPONENTS]
)
def test_the_component_conforms(component: type[Component], tmp_path: Path) -> None:
    assert conformance.check(component, tmp_path) == []


class _ChannelsOptions(Options):
    channels: int = Field(2, description="Channels to add.")


class DropsRows(Step):
    """Broken step: loses the last row."""

    name = "drops_rows"
    reads: ClassVar[dict[str, Col]] = {"scores": Col("float32", (1, 1, 1))}

    def __call__(self, batch: Batch) -> Batch:
        return {k: v[:-1] for k, v in batch.items()}


class IgnoresItsOption(Step):
    """Broken step: declares `channels` channels but always adds one."""

    name = "ignores_its_option"
    Options = _ChannelsOptions
    reads: ClassVar[dict[str, Col]] = {"scores": Col("float32", (1, 1, 1))}
    adds: ClassVar[dict[str, Col]] = {"probs": Col("float32", ("channels", 1, 1))}

    def __call__(self, batch: Batch) -> Batch:
        return batch | {"probs": batch["scores"]}


class FailsWhenEmpty(Sink):
    """Broken sink: writes nothing when no item came."""

    name = "fails_when_empty"
    format = "none"
    example_options: ClassVar[dict[str, Any]] = {"output": "x"}

    def open(self, frame: dict[str, Any], run: Any) -> None:
        self.path = run.work_dir / "out.npy"
        self.parts: list[np.ndarray] = []

    def add(self, batch: Batch) -> None:
        self.parts.append(np.zeros(1))

    def close(self) -> Path:
        if self.parts:
            np.save(self.path, np.concatenate(self.parts))
        return self.path


class _UndescribedOptions(Options):
    factor: float = 1.0


class Undocumented(Step):
    name = "undocumented"
    Options = _UndescribedOptions

    def __call__(self, batch: Batch) -> Batch:
        return batch


@pytest.mark.parametrize(
    ("component", "failure"),
    [
        (DropsRows, "step `drops_rows`: returned 0 rows for 1; a step keeps every row"),
        (
            IgnoresItsOption,
            "step `ignores_its_option`: `probs` is float32[1,1,1], but is declared "
            "as float32[2,1,1]",
        ),
        (
            FailsWhenEmpty,
            "sink `fails_when_empty`: wrote no artifact for no items",
        ),
        (Undocumented, "step `undocumented`: needs a docstring"),
        (Undocumented, "step `undocumented`: option `factor` needs a description"),
    ],
)
def test_the_suite_catches_a_broken_component(
    component: type[Component], failure: str, tmp_path: Path
) -> None:
    failures = conformance.check(component, tmp_path)

    assert any(f.startswith(failure) for f in failures), failures
