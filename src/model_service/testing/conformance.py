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
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from model_service.components.columns import MODEL_PREFIX, Col, rows
from model_service.components.kinds import Sink, Source, Step
from model_service.components.problems import Problem
from model_service.testing.models import fake_model
from model_service.testing.running import fake_batch, run_component
from model_service.testing.slides import fake_frame


if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

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
        except Exception as error:
            failures.append(f"raised {type(error).__name__}: {error}")
    return [f"{component.kind} `{component.name}`: {f}" for f in failures]


def _declaration(component: type[Component]) -> list[str]:
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
    failures = []
    for name, want in declared.items():
        if name not in batch:
            failures.append(f"declares it {what} `{name}` but the batch lacks it")
            continue
        have = Col(str(batch[name].dtype), tuple(batch[name].shape[1:]))
        if not want.fits(have):
            failures.append(f"`{name}` is {have}, but is declared as {want}")
    return failures
