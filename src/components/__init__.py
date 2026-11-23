"""Components: the small pieces of code that catalog files name by `use:`.

This package is what you write; `model_service` is the plumbing that runs it.
A component is a class deriving from its kind in `base/`, with a `name`, in
the folder that says where it may be used:

- `steps/`: a served model's `pre:`/`post:` and an analysis's `steps:`;
- `loaders/`: a served model's `weights:` (and an analysis step's own weights);
- `runtimes/`: inside a served model only;
- `sources/`, `sinks/`: inside an analysis only;
- `empaia/converters/`, `empaia/publishers/`: an analysis's EMPAIA exposure.

Also here: `base/`, the kind base classes and the types they use;
`_templates/`, one file per kind to copy into its folder; `testing/`, fakes and
the conformance suite for testing a component without a cluster.

`model_service.registry` scans the folders above and nothing else lists
components. Importing this package scans nothing.
"""
