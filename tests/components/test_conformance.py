"""Every registered component passes its kind's conformance suite.

The repo's registry is checked as it grows; the test-only components prove the
suites pass on components that behave, and the broken ones below prove they
catch the mistakes they exist for. A loader or runtime without an entry in
`helpers/offline.py` is skipped by name after its declaration passes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pytest
from pydantic import Field

from components.base import (
    Col,
    Commit,
    EmpaiaInput,
    EmpaiaOutput,
    InputConverter,
    Loader,
    Options,
    Publisher,
    Runtime,
    Sink,
    Step,
)
from model_service.registry import Registry, default_registry
from tests.components.helpers import conformance


if TYPE_CHECKING:
    from pathlib import Path

    from components.base import Batch, BatchFunction, Component, RuntimeContext


COMPONENTS = [*default_registry(), *Registry.scan("tests.support.components")]


@pytest.mark.parametrize(
    "component", COMPONENTS, ids=[f"{c.__module__}:{c.name}" for c in COMPONENTS]
)
def test_the_component_conforms(component: type[Component], tmp_path: Path) -> None:
    assert conformance.check(component, tmp_path) == []
    if conformance.declaration_only(component):
        pytest.skip(
            f"{component.kind} `{component.name}` has no entry in "
            "tests/components/helpers/offline.py; only its declaration was checked"
        )


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


class _PathOptions(Options):
    path: str = Field(description="The weights file.")


class LosesItsWeights(Loader):
    """Broken loader: fetches to a path that isn't there, and names no revision."""

    name = "loses_its_weights"
    Options = _PathOptions
    example_options: ClassVar[dict[str, Any]] = {"path": "/weights.bin"}

    def fetch(self, cache_dir: Path) -> Path:
        return cache_dir / "never-written.bin"

    def revision(self) -> str:
        return ""


class LosesARow(Runtime):
    """Broken runtime: its batch function drops the last tile."""

    name = "loses_a_row"

    def load(self, path: Path, context: RuntimeContext) -> BatchFunction:
        return lambda batch: np.zeros((len(batch) - 1, 1), dtype=np.float32)


class Untyped(InputConverter):
    """Broken converter: its EAD entry has no type."""

    name = "untyped"
    ead_type = "string"
    gives_file = False

    def io_fragment(self, item: EmpaiaInput) -> dict[str, Any]:
        return {"description": "no type"}

    async def materialize(
        self, gateway: Any, item: EmpaiaInput, directory: Path
    ) -> Any:
        return await gateway.get_input_data(item.name)


class PostsTooEarly(Publisher):
    """Broken publisher: posts during prepare, so a later failure can't undo it."""

    name = "posts_too_early"
    reads_format = "score_grid"
    emits = frozenset({"float"})
    example_options: ClassVar[dict[str, Any]] = {"type": "float"}

    def io_fragment(self, item: EmpaiaOutput) -> dict[str, Any]:
        return {"type": "float"}

    async def prepare(
        self, gateway: Any, item: EmpaiaOutput, artifact: Path, media_type: str
    ) -> Commit:
        await gateway.post_output(item.name, {})

        async def commit() -> None:
            pass

        return commit


@pytest.mark.parametrize(
    ("component", "failure"),
    [
        (
            LosesItsWeights,
            "loader `loses_its_weights`: revision() must return a non-empty string",
        ),
        (LosesItsWeights, "loader `loses_its_weights`: fetch returned "),
        (LosesARow, "runtime `loses_a_row`: the loaded function returned 1 rows for 2"),
        (Untyped, "converter `untyped`: io_fragment needs a string `type`"),
        (
            PostsTooEarly,
            "publisher `posts_too_early`: prepare wrote to EMPAIA (post_output)",
        ),
        (PostsTooEarly, "publisher `posts_too_early`: commit wrote nothing to EMPAIA"),
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
