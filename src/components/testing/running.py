"""Run one component in isolation, the way the runner would call it."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from components.base import Sink, Source, Step
from components.base.columns import Col
from components.base.run import Run
from components.testing.slides import fake_frame, fake_slide


if TYPE_CHECKING:
    from pathlib import Path

    from components.base.columns import Batch
    from components.base.component import Component


@dataclass(frozen=True)
class SourceRun:
    """What a source produced: its frame, keys and every item loaded.

    Attributes:
        frame: The frame row.
        keys: The item keys.
        batch: The items of all keys, after any the source dropped.
    """

    frame: dict[str, Any]
    keys: list[Any]
    batch: Batch


def fake_batch(columns: Mapping[str, Col], rows: int, seed: int = 0) -> Batch:
    """Random data for declared columns; unknown dimensions become 1.

    A column of any rank (`Col(dtype, None)`) gets one dimension.

    Args:
        columns: Resolved column declarations.
        rows: Rows to make.
        seed: Seed of the random values.

    Returns:
        A batch: uint8 columns take 0..255, float columns 0..1.
    """
    rng = np.random.default_rng(seed)
    batch: Batch = {}
    for name, col in columns.items():
        shape_ = (None,) if col.shape is None else col.shape
        dims = tuple(d if isinstance(d, int) else 1 for d in shape_)
        dtype = np.dtype(col.dtype)
        shape = (rows, *dims)
        if np.issubdtype(dtype, np.integer):
            batch[name] = rng.integers(0, 256, shape).astype(dtype)
        else:
            batch[name] = rng.random(shape).astype(dtype)
    return batch


def instance(
    component: type[Component] | Component, options: dict[str, Any] | None = None
) -> Component:
    """A component instance: given, or built from options or its example.

    Args:
        component: A component class or instance.
        options: Options for a class; its `example_options` by default.

    Returns:
        The component.
    """
    if isinstance(component, type):
        return component.from_options(
            component.example_options if options is None else options
        )
    return component


def run_component(
    component: type[Component] | Component,
    data: Any = None,
    *,
    options: dict[str, Any] | None = None,
    model: Any = None,
    work_dir: Path | None = None,
    frame: dict[str, Any] | None = None,
) -> Any:
    """Run a step, source or sink once, as the analysis runner would.

    Args:
        component: A component class (built from `options` or its example
            options) or instance.
        data: A step's batch, a sink's batch or list of batches, or a
            source's inputs by name. By default: random data for the declared
            reads, or a fake slide for each input.
        options: Options when `component` is a class.
        model: The served-model declaration for `model.` dimensions; a sink
            and source also get it as `run.model`.
        work_dir: Where sinks write and fake inputs go; needed for sources
            and sinks.
        frame: A sink's frame row; `fake_frame()` by default.

    Returns:
        A step's output batch, a sink's artifact path, or a source's
        `SourceRun`.

    Raises:
        TypeError: For kinds that are not run on batches.
        ValueError: If a source or sink gets no `work_dir`.
    """
    built = instance(component, options)
    if isinstance(built, Step):
        if data is None:
            data = fake_batch(built.columns(model)[0], rows=2)
        return built(dict(data))
    if not isinstance(built, Source | Sink):
        raise TypeError(
            f"run_component runs steps, sources and sinks, not a {built.kind}; "
            "call its methods directly"
        )
    if work_dir is None:
        raise ValueError(f"run_component needs work_dir to run a {built.kind}")
    if isinstance(built, Sink):
        reads = built.columns(model)[0]
        batches = [fake_batch(reads, 2)] if data is None else data
        batches = [batches] if isinstance(batches, Mapping) else list(batches)
        run = Run(work_dir, model, _known(reads, batches[0] if batches else {}))
        built.open(frame if frame is not None else fake_frame(), run)
        for batch in batches:
            built.add(dict(batch))
        return built.close()
    if data is None:
        data = {name: fake_slide(work_dir / f"{name}.tiff") for name in built.inputs()}
    with built.open(data, Run(work_dir, model)) as opened:
        keys = list(opened.keys)
        return SourceRun(opened.frame, keys, opened.load(keys))


def _known(reads: Mapping[str, Col], batch: Mapping[str, np.ndarray]) -> dict[str, Col]:
    """Fill unknown dimensions from a sample batch, or with 1 as `fake_batch` does.

    Args:
        reads: The columns a sink reads.
        batch: A sample batch, possibly empty.

    Returns:
        The columns with every dimension a number.
    """
    known = {}
    for name, col in reads.items():
        sample = batch[name].shape[1:] if name in batch else None
        shape = col.shape if col.shape is not None else sample or ()
        known[name] = Col(
            col.dtype,
            tuple(
                d if isinstance(d, int) else (sample[i] if sample else 1)
                for i, d in enumerate(shape)
            ),
        )
    return known
