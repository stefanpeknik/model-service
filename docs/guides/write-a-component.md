# Write a component

Catalog files name components with `{use: <name>, …}`. When the piece you
need doesn't exist, write it as one small class in one file, with one test.
You don't touch the catalog loader, the host or the runner: a folder scan
finds the new class, checks its declaration and lists it in the reference
docs.

## Pick the kind and the folder

Everything you write lives in `src/components/`. The kind decides where a
catalog file may name the component, and its folder there must match it:

| You need | Kind | Folder in `src/components/` |
| --- | --- | --- |
| Weights from somewhere new | loader | `loaders/` |
| A new way to run weights | runtime | `runtimes/` |
| A per-tile transform, in a model's `pre:`/`post:` or as a pipeline step | step | `steps/` |
| A new way to produce items from a slide | source | `sources/` |
| A new kind of output file | sink | `sinks/`, plus a module in `src/model_service/formats/` if the format is new |
| A new EMPAIA input or output type | converter, publisher | `providers/empaia/converters/`, `providers/empaia/publishers/` |

Each kind's base class in `src/components/_kinds/` names both: its `slots`
and its `folder`. A component in the wrong folder fails the scan with an
error naming the folder it belongs in.

Start from your kind's template in
[src/components/_templates/](../../src/components/_templates/):
[loader](../../src/components/_templates/loader.py),
[runtime](../../src/components/_templates/runtime.py),
[step](../../src/components/_templates/step.py),
[source](../../src/components/_templates/source.py),
[sink](../../src/components/_templates/sink.py),
[converter](../../src/components/_templates/converter.py) or
[publisher](../../src/components/_templates/publisher.py). Copy it into the
folder, then replace `my_component`, `MyComponent` and the `<placeholders>`.
Each template passes its kind's conformance suite as copied
(`tests/components/test_templates.py` checks that), so the suite's first
failure points at your change. The scan skips `_templates/`, so the
templates never show up as components.

Each kind's base class in `src/components/_kinds/` (`step.py`, `sink.py`, ...)
documents the methods to write, and [Components](../reference/components.md) links
the code of every existing component, often the closest example.

## Write it

Say you want a mask, not a score map: every score at or above a cutoff
becomes 1, the rest 0. A step in the model's `post:` does that.
`src/components/steps/threshold.py`:

```python
"""The `threshold` step: scores at or above a cutoff become 1, the rest 0."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from pydantic import Field

from components import Col, Options, Step


if TYPE_CHECKING:
    from components import Batch


class ThresholdOptions(Options):
    """Options of the `threshold` step."""

    at: float = Field(0.5, description="The cutoff; scores at or above it become 1.")


class Threshold(Step):
    """Turns scores into a 0/1 mask at a cutoff.

    `post: [{use: threshold, at: 0.65}]` after a model that returns
    probabilities gives 1 where the score reaches 0.65.
    """

    name = "threshold"
    Options = ThresholdOptions
    example_options: ClassVar[dict[str, Any]] = {"at": 0.5}
    reads: ClassVar[dict[str, Col]] = {"output": Col("float32", None)}
    adds: ClassVar[dict[str, Col]] = {"output": Col("float32", None)}
    options: ThresholdOptions

    def __call__(self, batch: Batch) -> Batch:
        """Compare every score with the cutoff.

        Args:
            batch: `output`, scores of any shape.

        Returns:
            The batch with 0 or 1 in `output`.
        """
        mask = batch["output"] >= self.options.at
        return batch | {"output": mask.astype(np.float32)}
```

The shared steps in `src/components/steps/` are real examples of the same
shape: `normalize.py`, `resize.py` and `center_crop.py` prepare tiles, and
`cls_mean_pool.py` reduces a transformer's tokens to embeddings.

What each part is for:

- `name` is what catalog files write after `use:`. Two components of one
  kind can't share a name.
- `Options` lists what a file may set. Unknown keys are errors, and every
  field needs a `description`, which becomes the option's line in the
  reference docs and the editor's hover text. Mark an option `PerRun[...]`
  if callers may change it for one run; it then becomes a pipeline setting
  and an EMPAIA job parameter.
- `reads` and `adds` declare the columns, as dtype and shape without the
  batch axis. A dimension is a number, `None` for any size, or the name of
  one of your options. The catalog check uses them to tell a file author that your step
  doesn't fit before anything runs. A model's runtime output is the
  column `output`; in a pipeline, a model step adds `scores` or `features`.
- The class docstring is the component's reference page.
- Import heavy libraries (torch, ONNX Runtime, OpenSlide) inside methods,
  never at the top of the module. The control plane imports every component to check
  catalog files and must stay free of them.
- Override `hint()` if you can tell a file author which value to change when
  a check fails; `huggingface.py` does it for a bad `revision`.

## Test it

`tests/components/test_conformance.py` runs your kind's conformance suite on
every registered component, so yours is covered as soon as the file exists.
It checks the docstring, the option descriptions, `example_options` and the
declared dimensions, then runs the component the way its kind is used:

- a step keeps every row and adds the columns it declares;
- a source gives its frame fields and loads its declared columns from a
  fake slide; a sink writes a file with no items and with some;
- a loader fetches a local weights file to a path that exists, and its
  `revision()` is a non-empty string that matches `expected_revision()`;
- a runtime loads tiny weights into a function that keeps the batch length;
- a converter's `io_fragment` has a string `type`, and `materialize` reads
  a fake EMPAIA job (`FakeGateway`) into a file in its directory or a JSON
  value without writing to the job;
- a publisher's `io_fragment` has a `type` it emits, `prepare` on a sample
  file of its format writes nothing, and the commit it returns writes.

A loader or a runtime needs an entry, keyed by its name, in
`tests/components/helpers/offline.py`: for a loader, options that fetch a
given local file without a network (`local` points `path` at it, `mlflow`
gives a `file://` URI); for a runtime, a function that writes tiny weights
it loads and a batch for them (`onnx` gets a two-node graph, `timm` timm's
`test_vit`, `constant` an empty file). Without one, the suite checks only
the declaration and the test is skipped with the component's name.

A converter for a new EAD type adds example input data to
`EXAMPLE_INPUTS`, and a publisher of a new format adds a sample writer to
`SAMPLES`, both in `tests/components/helpers/empaia.py`.

Add a test of what your component computes. `tests.components.helpers` runs
a component the way the runner would, without Ray or a cluster:

```python
import numpy as np

from components.steps.threshold import Threshold
from tests.components.helpers import run_component


def test_threshold_keeps_scores_at_or_above_the_cutoff() -> None:
    scores = np.array([[0.2, 0.5, 0.9]], dtype=np.float32)

    out = run_component(Threshold, {"output": scores}, options={"at": 0.5})

    np.testing.assert_array_equal(out["output"], [[0.0, 1.0, 1.0]])
```

The other helpers cover the rest of the kinds:

- `run_component(MySource, work_dir=tmp_path)` writes a `fake_slide()` for
  each input the source takes and returns its frame, keys and loaded items.
- `run_component(MySink, batches, work_dir=tmp_path)` feeds a sink batches
  and a `fake_frame()` and returns the artifact it wrote.
- `fake_model(output={"kind": "embedding", "summaries": ["global"]})` stands
  in for a model, for `model.` dimensions and model calls.

Regenerate the reference docs so they list your component, then run the
whole suite:

```sh
PYTHONPATH=src uv run tools/generate.py
uv run pytest
```

Run `uv run pytest` before every deploy. The catalog check, the conformance
suite and your own test are what keep a broken component or catalog file
off the cluster; Helm refuses only a few cases when it renders, and nothing
else checks the file before a replica imports it.

## Use it

Name it in a catalog file and check that file:
`post: [{use: threshold}]` in a model, then `uv run pytest tests/catalog`.
A component is code, so deploying it needs a new image; catalog files that
use it deploy with `helm upgrade` once the image runs.
