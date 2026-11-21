"""Run one component in isolation, the way the runner would call it."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from model_service.components.kinds import Sink, Source, Step
from model_service.testing.slides import fake_frame, fake_slide


if TYPE_CHECKING:
    from pathlib import Path

    from model_service.components.columns import Batch, Col
    from model_service.components.component import Component


@dataclass(frozen=True)
class FakeRun:
    """The run context sources and sinks receive.

    Attributes:
        work_dir: Where sinks write their artifacts.
        model: The run's served model, if any.
    """

    work_dir: Path
    model: Any = None


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
        dims = tuple(d if isinstance(d, int) else 1 for d in col.shape or ())
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
    run = FakeRun(work_dir, model)
    if isinstance(built, Sink):
        batches = [fake_batch(built.columns(model)[0], 2)] if data is None else data
        built.open(frame if frame is not None else fake_frame(), run)
        for batch in [batches] if isinstance(batches, Mapping) else batches:
            built.add(dict(batch))
        return built.close()
    if data is None:
        data = {name: fake_slide(work_dir / f"{name}.tiff") for name in built.inputs()}
    with built.open(data, run) as opened:
        keys = list(opened.keys)
        return SourceRun(opened.frame, keys, opened.load(keys))
