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

from components._kinds.columns import Batch, Col, Dim, rows
from components._kinds.component import Component
from components._kinds.converter import EmpaiaInput, InputConverter
from components._kinds.loader import Loader
from components._kinds.options import Options, PerRun, per_run_fields, with_overrides
from components._kinds.problems import Problem
from components._kinds.publisher import (
    Commit,
    EmpaiaOutput,
    Publisher,
    PublisherOptions,
)
from components._kinds.run import Run
from components._kinds.runtime import (
    BatchFunction,
    Runtime,
    RuntimeContext,
)
from components._kinds.sink import Sink, SinkOptions
from components._kinds.slots import Slot
from components._kinds.source import Opened, Source, TileGeometryOptions
from components._kinds.step import Step


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
