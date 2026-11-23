"""Conformance suites: what every component of a kind must do.

`check(component, work_dir)` runs the suite of the component's kind and
returns what failed, so a test asserts it is empty. The repo's tests run it
for every registered component; a newcomer's component is covered as soon as
it is in its folder.

Every kind: a docstring, a description on every option, valid
`example_options`, symbolic dimensions that name an option or `model.<field>`,
`may_add` covering `adds`, and a `hint()` that answers.
Steps: keep the batch length and add the declared columns as declared.
Sinks: read only frame fields a slide source gives, and write an artifact
with zero items and with some.
Sources: give the declared frame fields and load the declared columns.
Loaders: name a revision; with `offline_options`, fetch a stand-in file to a
path that exists and report the revision they promise.
Runtimes: load their `stand_in()` weights into a function that keeps the
batch length, and report providers as strings.
Converters: give an EAD entry with a `type`, and materialise a fake job's
input as a file in their directory or as a JSON value, writing nothing.
Publishers: give an EAD entry of a type they emit; `prepare` on a sample file
writes nothing to the fake job, and its commit writes.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import ValidationError

from model_service.components.columns import MODEL_PREFIX, Col, rows
from model_service.components.kinds import (
    EmpaiaInput,
    EmpaiaOutput,
    InputConverter,
    Loader,
    Publisher,
    Runtime,
    RuntimeContext,
    Sink,
    Source,
    Step,
)
from model_service.components.problems import Problem
from model_service.testing.empaia import FakeGateway, sample_artifact
from model_service.testing.models import fake_model
from model_service.testing.running import fake_batch, run_component
from model_service.testing.slides import fake_frame


if TYPE_CHECKING:
    from collections.abc import Mapping

    from model_service.components.columns import Batch
    from model_service.components.component import Component


def check(component: type[Component], work_dir: Path) -> list[str]:
    """Run the conformance suite of the component's kind.

    Args:
        component: A component class.
        work_dir: An empty directory for artifacts and fake inputs.

    Returns:
        One sentence per failed check; empty when the component conforms.
    """
    failures = _declaration(component)
    if not failures:
        built = component.from_options(component.example_options)
        try:
            if isinstance(built, Step):
                failures += _step(built)
            elif isinstance(built, Sink):
                failures += _sink(component, work_dir)
            elif isinstance(built, Source):
                failures += _source(built, work_dir)
            elif isinstance(built, Loader):
                failures += _loader(built, work_dir)
            elif isinstance(built, Runtime):
                failures += _runtime(built, work_dir)
            elif isinstance(built, InputConverter):
                failures += _converter(built, work_dir)
            elif isinstance(built, Publisher):
                failures += _publisher(built, work_dir)
        except Exception as error:
            failures.append(f"raised {type(error).__name__}: {error}")
    return [f"{component.kind} `{component.name}`: {f}" for f in failures]


def _declaration(component: type[Component]) -> list[str]:
    """Check what every kind declares: docstring, options, dimensions, hint.

    Args:
        component: The component class.

    Returns:
        What failed.
    """
    failures = []
    if not (component.__doc__ or "").strip() or "__doc__" not in component.__dict__:
        failures.append("needs a docstring; it becomes its reference page")
    for name, field in component.Options.model_fields.items():
        if not field.description:
            failures.append(
                f"option `{name}` needs a description (Field(description=…))"
            )
    try:
        component.Options.model_validate(component.example_options)
    except ValidationError as error:
        failures.append(f"example_options do not validate: {error}")
    columns = {**component.reads, **component.adds}
    if issubclass(component, Source):
        columns |= component.frame
    for column, col in columns.items():
        for dim in col.shape or ():
            if (
                isinstance(dim, str)
                and not dim.startswith(MODEL_PREFIX)
                and dim not in component.Options.model_fields
            ):
                failures.append(
                    f"column `{column}` has dimension `{dim}`, which is neither an "
                    "option nor `model.<field>`"
                )
    missing = sorted(set(component.adds) - component.may_add)
    if missing:
        failures.append(f"may_add lacks columns it adds: {missing}")
    hint = component.hint(Problem("conformance", "a check failed", "something else"))
    if hint is not None and not isinstance(hint, str):
        failures.append("hint() must return a string or None")
    return failures


def _step(step: Step) -> list[str]:
    """Check a step keeps every row and adds its columns as declared.

    Args:
        step: The step, built from its example options.

    Returns:
        What failed.
    """
    model = fake_model()
    reads, adds = step.columns(model)
    failures = []
    for n in (1, 3):
        out = run_component(step, fake_batch(reads, n), model=model)
        if rows(out) != n:
            failures.append(
                f"returned {rows(out)} rows for {n}; a step keeps every row"
            )
        failures += _columns_match(adds, out, "adds")
    return failures


def _sink(component: type[Component], work_dir: Path) -> list[str]:
    """Check a sink reads slide frame fields and writes a file with or without items.

    Args:
        component: The sink class.
        work_dir: An empty directory for its files.

    Returns:
        What failed with the first model output kind tried, or nothing.
    """
    assert issubclass(component, Sink)
    unknown = sorted(component.frame_reads - fake_frame().keys())
    if unknown:
        return [f"frame_reads {unknown} are not fields a slide source gives"]
    # A sink may need one model output kind (an embedding table needs
    # embeddings), so it conforms if it works with either.
    first: list[str] = []
    for kind in ("scores", "embedding"):
        failures = _sink_with(component, work_dir / kind, fake_model({"kind": kind}))
        if not failures:
            return []
        first = first or failures
    return first


def _sink_with(component: type[Sink], work_dir: Path, model: Any) -> list[str]:
    """Run a sink with no items and with some, and check it wrote an artifact.

    Args:
        component: The sink class.
        work_dir: A directory to create the runs' directories in.
        model: The stand-in served model the run gets.

    Returns:
        What failed.
    """
    failures = []
    for label, batches in (("no items", []), ("two items", None)):
        directory = work_dir / label.replace(" ", "-")
        directory.mkdir(parents=True)
        try:
            path = run_component(component, batches, model=model, work_dir=directory)
        except Exception as error:
            failures.append(f"raised {type(error).__name__}: {error}")
            continue
        if not path.exists():
            failures.append(f"wrote no artifact for {label} ({path} is missing)")
    return failures


def _source(source: Source, work_dir: Path) -> list[str]:
    """Check a source gives its frame fields and loads its declared columns.

    Args:
        source: The source, built from its example options.
        work_dir: An empty directory for its fake inputs.

    Returns:
        What failed.
    """
    result = run_component(source, work_dir=work_dir)
    failures = [
        f"the frame lacks the declared field `{name}`"
        for name in source.frame
        if name not in result.frame
    ]
    if rows(result.batch) > len(result.keys):
        failures.append(f"loaded {rows(result.batch)} rows for {len(result.keys)} keys")
    return failures + _columns_match(source.columns(None)[1], result.batch, "adds")


def _columns_match(declared: Mapping[str, Col], batch: Batch, what: str) -> list[str]:
    """Check a batch has each declared column, in a shape that fits.

    Args:
        declared: The declared columns.
        batch: The batch.
        what: How the component declares them, e.g. `adds`.

    Returns:
        What failed.
    """
    failures = []
    for name, want in declared.items():
        if name not in batch:
            failures.append(f"declares it {what} `{name}` but the batch lacks it")
            continue
        have = Col(str(batch[name].dtype), tuple(batch[name].shape[1:]))
        if not want.fits(have):
            failures.append(f"`{name}` is {have}, but is declared as {want}")
    return failures


def _text(value: object) -> bool:
    """Whether a value is a non-empty string.

    Args:
        value: Anything.

    Returns:
        True for a non-empty string.
    """
    return isinstance(value, str) and bool(value)


def _loader(loader: Loader, work_dir: Path) -> list[str]:
    """Check a loader names its weights and, offline, fetches them.

    Args:
        loader: The loader, built from its example options.
        work_dir: An empty directory for the stand-in weights.

    Returns:
        What failed.
    """
    failures = []
    if not _text(loader.revision()):
        failures.append("revision() must return a non-empty string naming the weights")
    weights = work_dir / "weights.bin"
    weights.write_bytes(b"stand-in weights")
    options = type(loader).offline_options(weights)
    if options is None:
        return failures
    offline = type(loader).from_options(options)
    assert isinstance(offline, Loader)
    cache = work_dir / "cache"
    cache.mkdir()
    fetched = offline.fetch(cache)
    if not isinstance(fetched, Path) or not fetched.exists():
        failures.append(f"fetch returned {fetched!r}, which does not exist")
    revision = offline.revision()
    if not _text(revision):
        failures.append("revision() must return a non-empty string after fetch")
    expected = offline.expected_revision()
    if expected is not None and expected != revision:
        failures.append(
            f"expected_revision() is {expected!r}, but revision() after fetch "
            f"is {revision!r}"
        )
    return failures


def _runtime(runtime: Runtime, work_dir: Path) -> list[str]:
    """Check a runtime loads its stand-in weights into a batch function.

    Args:
        runtime: The runtime, built from its example options.
        work_dir: An empty directory for the stand-in and its cache.

    Returns:
        What failed.
    """
    directory = work_dir / "stand-in"
    directory.mkdir()
    stand_in = type(runtime).stand_in(directory)
    batch = stand_in.batch
    run = runtime.load(
        stand_in.weights,
        RuntimeContext(
            cache_dir=work_dir / "cache",
            tile_size=int(batch.shape[-1]),
            max_batch_size=len(batch),
        ),
    )
    if not callable(run):
        return [f"load returned {run!r}, not a function of a batch"]
    output = np.asarray(run(batch))
    failures = []
    if output.ndim == 0 or len(output) != len(batch):
        got = 0 if output.ndim == 0 else len(output)
        failures.append(
            f"the loaded function returned {got} rows for {len(batch)}; "
            "it maps each tile of a batch to one output"
        )
    providers = runtime.active_providers()
    if not isinstance(providers, list) or not all(_text(p) for p in providers):
        failures.append(f"active_providers() returned {providers!r}, not names")
    return failures


def _converter(converter: InputConverter, work_dir: Path) -> list[str]:
    """Check a converter's EAD entry and its input from a fake job.

    Args:
        converter: The converter, built from its example options.
        work_dir: An empty directory for the fake job and the input.

    Returns:
        What failed.
    """
    item = EmpaiaInput(
        "input",
        slide="slide",
        setting={"type": "number", "description": "A setting."},
    )
    failures = []
    fragment = converter.io_fragment(item)
    if not isinstance(fragment, dict) or not _text(fragment.get("type")):
        failures.append(
            f"io_fragment needs a string `type`, e.g. {{'type': 'wsi'}}; got {fragment!r}"
        )
    gateway = FakeGateway(work_dir, input_types={item.name: converter.ead_type})
    directory = work_dir / "input"
    directory.mkdir()
    value = asyncio.run(converter.materialize(gateway, item, directory))
    if converter.gives_file:
        if (
            not isinstance(value, Path)
            or value.parent != directory
            or not value.exists()
        ):
            failures.append(
                f"materialize returned {value!r}; with gives_file it writes a "
                "file in the directory it is given"
            )
    else:
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            failures.append(
                f"materialize returned {value!r}; without gives_file it gives "
                "a JSON value"
            )
    if gateway.writes:
        failures.append("materialize wrote to EMPAIA; a converter only reads")
    return failures


def _publisher(publisher: Publisher, work_dir: Path) -> list[str]:
    """Check a publisher's EAD entry and its prepare-then-commit on a fake job.

    Args:
        publisher: The publisher, built from its example options.
        work_dir: An empty directory for the fake job and the sample file.

    Returns:
        What failed.
    """
    item = EmpaiaOutput("output", slide="slide")
    failures = []
    fragment = publisher.io_fragment(item)
    if not isinstance(fragment, dict) or fragment.get("type") not in publisher.emits:
        failures.append(
            f"io_fragment's `type` must be one it emits ({sorted(publisher.emits)}); "
            f"got {fragment!r}"
        )
    sample = sample_artifact(publisher.reads_format, work_dir)
    if sample is None:
        return [
            *failures,
            f"model_service.testing.empaia.SAMPLES has no `{publisher.reads_format}` "
            "file to publish; add a writer for the format",
        ]
    artifact, media_type = sample
    gateway = FakeGateway(work_dir)

    async def publish() -> list[tuple[str, str]]:
        """Prepare the publication, note the writes so far, then commit.

        Returns:
            The writes `prepare` made.
        """
        commit = await publisher.prepare(gateway, item, artifact, media_type)
        prepared = list(gateway.writes)
        await commit()
        return prepared

    prepared = asyncio.run(publish())
    if prepared:
        calls = ", ".join(call for call, _ in prepared)
        failures.append(f"prepare wrote to EMPAIA ({calls}); only its commit may")
    if len(gateway.writes) == len(prepared):
        failures.append("commit wrote nothing to EMPAIA")
    return failures
