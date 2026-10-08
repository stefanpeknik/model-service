# Reading source docstrings

Python modules outside `models/` have module, class, and function docstrings,
including private helpers and test fixtures. The format follows the repository's
Google-style Ruff configuration.

Start with the module header to understand its responsibility and dependencies.
Read a class docstring for its role, then the method contract before tracing the
implementation. Type annotations describe the shape of a value; docstrings
explain its meaning, units, ownership, and lifecycle.

## Contract sections

- `Args` describes every explicit parameter except `self` and `cls`. It includes
  optional values, defaults that affect behavior, timing units, path boundaries,
  and resource ownership where relevant.
- `Returns` describes the returned value, including what `None` and boolean
  outcomes mean. Constructors describe initialization instead of returning a
  value. Methods that complete through side effects explicitly return `None`.
- `Yields` replaces `Returns` for generators and context managers. For example,
  `ExecutionScheduler.running()` yields control while its worker tasks run.
- `Raises` names errors that escape the documented boundary and explains their
  triggers. Errors caught and translated internally belong to the contract of
  the layer that exposes them. A function with no exceptional contract does not
  need an empty `Raises` section. Injected callbacks and native libraries can
  propagate their own errors; their relevant boundaries say so explicitly.
- `Attributes` explains fields on records, request models, and domain values.
  Derived classes document their added fields; inherited fields are documented
  on their base class.

Function docstrings also explain side effects that a return type cannot convey,
such as committing a transaction, fencing a stale attempt, publishing a provider
output, closing an injected client, or cancelling a local task.

## Follow a request

Use the existing package maps to choose a path through source:

| Question | Starting point |
| --- | --- |
| Which public route accepts this request? | [Public application map](../../model_service/README.md), then `model_service/api.py` |
| How does admission become leased background work? | [Scheduler map](../../execution_scheduler/README.md), then `scheduler.py`, `worker.py`, and `transitions.py` |
| Where does a Python call become HTTP dispatch? | `workflow_execution/dispatch.py`, which sends the internal invocation envelope to Ray Serve |
| How are inputs acquired and outputs checked? | [Workflow runtime map](../../workflow_execution/README.md), then `application.py` and the selected access adapter |
| How does a slide become a heatmap? | [Built-in workflow map](../../workflows/README.md), then `heatmap/prostate.py`, `processor.py`, `tile_reader.py`, and `ray_predictor.py` |
| Which behavior does a test establish? | Its test method docstring, followed by its fixtures and assertions |

Docstrings describe source behavior. Deployment records establish which revision
ran in which containers. See [the documentation entry point](../README.md) for
those records and their validation boundaries. The SQLite control plane under
`prototypes/` is historical and is labelled as such in its module headers.

## Maintaining docstrings

Update the contract when behavior changes. Document concrete exception triggers
and the meaning of optional results rather than copying a generic description
from a similarly named helper. Keep argument names synchronized with signatures.
Use `uv run ruff format` and `uv run ruff check --no-fix` for the changed files;
Ruff checks formatting, while source review must establish semantic accuracy.
