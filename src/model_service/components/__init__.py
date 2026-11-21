"""Components: small reusable pieces of code that catalog files name.

There are seven kinds (`Loader`, `Runtime`, `Step`, `Source`, `Sink`,
`InputConverter`, `Publisher`). A component is a class deriving from its kind
with a `name`, in the folder that says where it may be used:

- `served_model/loaders/`, `served_model/runtimes/`: inside a served model;
- `steps/`: a served model's `pre:`/`post:` and analysis steps;
- `analysis/sources/`, `analysis/sinks/`: inside an analysis;
- `empaia/input_converters/`, `empaia/publishers/`, `standalone/`: exposure.

`default_registry()` scans those folders; nothing else lists components.
Importing this package scans nothing, so a component module can import its
kind from here.
"""

from model_service.components.columns import Batch, Col, Dim, rows
from model_service.components.component import Component
from model_service.components.kinds import (
    KINDS,
    BatchFunction,
    Commit,
    InputConverter,
    Loader,
    Opened,
    Publisher,
    Runtime,
    Sink,
    SinkOptions,
    Source,
    Step,
)
from model_service.components.options import (
    Options,
    PerRun,
    per_run_fields,
    with_overrides,
)
from model_service.components.problems import Problem
from model_service.components.registry import Registry, default_registry
from model_service.components.slots import Slot


__all__ = [
    "KINDS",
    "Batch",
    "BatchFunction",
    "Col",
    "Commit",
    "Component",
    "Dim",
    "InputConverter",
    "Loader",
    "Opened",
    "Options",
    "PerRun",
    "Problem",
    "Publisher",
    "Registry",
    "Runtime",
    "Sink",
    "SinkOptions",
    "Slot",
    "Source",
    "Step",
    "default_registry",
    "per_run_fields",
    "rows",
    "with_overrides",
]
