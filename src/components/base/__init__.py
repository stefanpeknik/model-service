"""What a component author subclasses and uses: the kinds and their parts.

- One base class per kind, one module each: `Loader` (loader.py), `Runtime`
  (runtime.py), `Step` (step.py), `Source` (source.py), `Sink` (sink.py),
  `InputConverter` (converter.py), `Publisher` (publisher.py). They share
  `Component` (component.py).
- options.py: `Options`, the base of a component's options, and `PerRun[...]`
  for a field a caller may override for one run.
- columns.py: `Col` and `Dim`, the columns a component reads and adds; a batch
  is `dict[str, ndarray]`.
- run.py: `Run`, what a source or sink is told about its run.
- problems.py: `Problem`, the failed check a component's `hint()` explains.
- slots.py: the catalog keys each kind may be named in.

Nothing here imports `model_service` at run time, so a component module can
import its kind without loading the service.
"""

from components.base.columns import Batch, Col, Dim, rows
from components.base.component import Component
from components.base.converter import EmpaiaInput, InputConverter
from components.base.loader import Loader
from components.base.options import Options, PerRun, per_run_fields, with_overrides
from components.base.problems import Problem
from components.base.publisher import (
    Commit,
    EmpaiaOutput,
    Publisher,
    PublisherOptions,
)
from components.base.run import Run
from components.base.runtime import (
    BatchFunction,
    Runtime,
    RuntimeContext,
)
from components.base.sink import Sink, SinkOptions
from components.base.slots import Slot
from components.base.source import Opened, Source, TileGeometryOptions
from components.base.step import Step


KINDS = (Loader, Runtime, Step, Source, Sink, InputConverter, Publisher)

__all__ = [
    "KINDS",
    "Batch",
    "BatchFunction",
    "Col",
    "Commit",
    "Component",
    "Dim",
    "EmpaiaInput",
    "EmpaiaOutput",
    "InputConverter",
    "Loader",
    "Opened",
    "Options",
    "PerRun",
    "Problem",
    "Publisher",
    "PublisherOptions",
    "Run",
    "Runtime",
    "RuntimeContext",
    "Sink",
    "SinkOptions",
    "Slot",
    "Source",
    "Step",
    "TileGeometryOptions",
    "per_run_fields",
    "rows",
    "with_overrides",
]
