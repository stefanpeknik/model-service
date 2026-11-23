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
| A per-tile transform, in a served model's `pre:`/`post:` or as an analysis step | step | `steps/` |
| A new way to produce items from a slide | source | `sources/` |
| A new kind of output file | sink | `sinks/`, plus a module in `src/model_service/formats/` if the format is new |
| A new EMPAIA input or output type | converter, publisher | `empaia/converters/`, `empaia/publishers/` |

A component in the wrong folder fails the scan with an error naming the
folder it belongs in.

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

Each kind's base class in `src/components/base/` (`step.py`, `sink.py`, ...)
documents the methods to write, and [Components](../reference/components.md) links
the code of every existing component, often the closest example.

## Write it

Say a model returns logits and you want probabilities. A step in the served
model's `post:` fixes that. `src/components/steps/softmax.py`:

```python
"""The `softmax` step: turns a model's logits into probabilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from pydantic import Field

from components.base import Col, Options, Step


if TYPE_CHECKING:
    from components.base import Batch


class SoftmaxOptions(Options):
    """Options of the `softmax` step."""

    temperature: float = Field(
        1.0, gt=0, description="Divides the logits first; above 1 flattens the scores."
    )


class Softmax(Step):
    """Turns per-channel logits into probabilities that sum to 1.

    `post: [{use: softmax}]` fixes a model that returns logits, so its
    file can declare `output: {range: [0, 1]}`.
    """

    name = "softmax"
    Options = SoftmaxOptions
    example_options: ClassVar[dict[str, Any]] = {"temperature": 1.0}
    reads: ClassVar[dict[str, Col]] = {"output": Col("float32", (None, None, None))}
    adds: ClassVar[dict[str, Col]] = {"output": Col("float32", (None, None, None))}
    options: SoftmaxOptions

    def __call__(self, batch: Batch) -> Batch:
        """Apply softmax over the channel axis.

        Args:
            batch: `output`, logits `(rows, channels, size, size)`.

        Returns:
            The batch with probabilities in `output`.
        """
        logits = batch["output"] / self.options.temperature
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        return batch | {"output": exp / exp.sum(axis=1, keepdims=True)}
```

What each part is for:

- `name` is what catalog files write after `use:`. Two components of one
  kind can't share a name.
- `Options` lists what a file may set. Unknown keys are errors, and every
  field needs a `description`, which becomes the option's line in the
  reference docs and the editor's hover text. Mark an option `PerRun[...]`
  if callers may change it for one run; it then becomes an analysis setting
  and an EMPAIA job parameter.
- `reads` and `adds` declare the columns, as dtype and shape without the
  batch axis. A dimension is a number, `None` for any size, the name of one
  of your options, or `model.input.tile_size`-style fields of the served
  model. The catalog check uses them to tell a file author that your step
  doesn't fit before anything runs. A served model's runtime output is the
  column `output`; in an analysis, a model step adds `scores` or `features`.
- The class docstring is the component's reference page.
- Import heavy libraries (torch, ONNX Runtime, OpenSlide) inside methods,
  never at the top of the module. The API imports every component to check
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
- a loader's `revision()` is a non-empty string. If it can fetch without a
  network, return options that do from `offline_options(weights)`
  (`local.py` and `mlflow.py` do), and the suite checks that `fetch`
  returns a path that exists and that `revision()` matches
  `expected_revision()`;
- a runtime loads the weights its `stand_in(directory)` writes into a
  function that keeps the batch length. The default stand-in is an empty
  file and two uint8 tiles; a runtime of a weights format writes a tiny
  model (`onnx.py` writes a two-node graph, `timm.py` timm's `test_vit`);
- a converter's `io_fragment` has a string `type`, and `materialize` reads
  a fake EMPAIA job (`components.testing.FakeGateway`) into a file in
  its directory or a JSON value without writing to the job;
- a publisher's `io_fragment` has a `type` it emits, `prepare` on a sample
  file of its format writes nothing, and the commit it returns writes.

A converter for a new EAD type adds example input data to
`EXAMPLE_INPUTS`, and a publisher of a new format adds a sample writer to
`SAMPLES`, both in `src/components/testing/empaia.py`.

Add a test of what your component computes. `components.testing` runs a
component the way the runner would, without Ray or a cluster:

```python
import numpy as np

from components.steps.softmax import Softmax
from components.testing import run_component


def test_softmax_gives_probabilities_that_sum_to_one() -> None:
    logits = np.array([[[[2.0]], [[0.0]], [[-1.0]]]], dtype=np.float32)

    out = run_component(Softmax, {"output": logits})

    np.testing.assert_allclose(out["output"].sum(axis=1), 1.0, rtol=1e-6)
    assert out["output"][0, 0, 0, 0] > out["output"][0, 1, 0, 0]
```

The other helpers cover the rest of the kinds:

- `run_component(MySource, work_dir=tmp_path)` writes a `fake_slide()` for
  each input the source takes and returns its frame, keys and loaded items.
- `run_component(MySink, batches, work_dir=tmp_path)` feeds a sink batches
  and a `fake_frame()` and returns the artifact it wrote.
- `fake_model(output={"kind": "embedding", "summaries": ["global"]})` stands
  in for a served model, for `model.` dimensions and model calls.

Run the tests, then regenerate the reference docs so they list your
component:

```sh
uv run pytest tests/components
PYTHONPATH=src uv run tools/generate.py
```

## Use it

Name it in a catalog file and check that file:
`post: [{use: softmax}]` in a served model, then `uv run pytest tests/catalog`.
A component is code, so deploying it needs a new image; catalog files that
use it deploy with `helm upgrade` once the image runs.
