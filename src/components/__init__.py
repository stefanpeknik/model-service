"""Components: the small pieces of code that catalog files name by `use:`.

This package is what you write; `model_service` is the plumbing that runs it.
A component is a class deriving from its kind, imported from here
(`from components import Step, Options, Col`), with a `name`, in the folder
that says where it may be used:

- `steps/`: a served model's `pre:`/`post:` and an analysis's `steps:`;
- `loaders/`: a served model's `loader:` (and an analysis step's own loader);
- `runtimes/`: inside a served model only;
- `sources/`, `sinks/`: inside an analysis only;
- `providers/empaia/converters/`, `providers/empaia/publishers/`: an
  analysis's EMPAIA settings under `providers.empaia`.

Also here: `_kinds/`, the kind base classes and the types they use (import
them from this package, not from there); `_templates/`, one file per kind to
copy into its folder. The helpers for testing a component without a cluster
are test code, in `tests/components/helpers/`.

`model_service.registry` scans the folders above and nothing else lists
components. Importing this package scans nothing.
"""

from components._kinds import (
    KINDS,
    Batch,
    BatchFunction,
    Col,
    Commit,
    Component,
    Dim,
    EmpaiaInput,
    EmpaiaOutput,
    InputConverter,
    Loader,
    Opened,
    Options,
    PerRun,
    Problem,
    Publisher,
    PublisherOptions,
    Run,
    Runtime,
    RuntimeContext,
    Sink,
    SinkOptions,
    Slot,
    Source,
    Step,
    TileGeometryOptions,
    per_run_fields,
    rows,
    slots,
    with_overrides,
)


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
    "slots",
    "with_overrides",
]
