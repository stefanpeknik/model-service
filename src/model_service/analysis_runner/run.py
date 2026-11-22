"""The analysis runner core: one resolved analysis, run on one set of inputs.

Each key the source gives is one task: the source loads its item in the read
pool, the steps run in order (model steps await the served model, component
steps run in the pool), and the rows go to the sinks. Rows reach unordered
sinks as each task finishes, as today's heatmap loop updates its grid, and in
key order when any sink is ordered. A key whose item the source drops (a
background tile) still counts toward progress.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from model_service.analysis_runner.models import model_call
from model_service.analysis_runner.outputs import Staging, check_destinations
from model_service.analysis_runner.timing import Timings
from model_service.catalog.analyses import ModelStep
from model_service.components import Run, rows
from model_service.contracts.failures import (
    BuilderFailure,
    FailureCode,
    InvalidInvocation,
)


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence

    from model_service.analysis_runner.models import ModelHandle
    from model_service.catalog.analyses import Analysis
    from model_service.components import Batch, Sink


logger = logging.getLogger(__name__)
PROGRESS_LOG_SECONDS = 60


@dataclass(frozen=True)
class Concurrency:
    """How much of one run happens at once; the defaults are today's builders'.

    Attributes:
        read_threads: Threads for source loads and component steps.
        in_flight: Keys being processed at once.
    """

    read_threads: int = 2
    in_flight: int = 4

    def __post_init__(self) -> None:
        """Refuse limits below one.

        Raises:
            ValueError: If a limit is below one.
        """
        if self.read_threads < 1 or self.in_flight < 1:
            raise ValueError("read_threads and in_flight must be at least 1")


def _ignore(_: float) -> None:
    """Drop a progress report."""


async def run_analysis(
    analysis: Analysis,
    inputs: Mapping[str, Path],
    outputs: Mapping[str, Path],
    models: Mapping[str, ModelHandle],
    *,
    concurrency: Concurrency = Concurrency(),  # noqa: B008 - frozen dataclass
    report_progress: Callable[[float], None] = _ignore,
) -> dict[str, Path]:
    """Run an analysis and write its outputs, all of them or none.

    Cancelling the calling task stops the run; like any failure, it leaves no
    output and no staging file behind, and every destination as it was.

    Args:
        analysis: The resolved analysis, with the run's settings applied.
        inputs: Input name to local path.
        outputs: Output name (a sink's `output:`) to the path to write.
        models: Served-model ID to its handle, for every model step.
        concurrency: Read threads and keys in flight.
        report_progress: Takes the share of the source's keys done, 0 to 1;
            called once per key, so it must be cheap.

    Returns:
        Output name to the written path.

    Raises:
        InvalidInvocation: For missing, unknown or unreadable inputs, or
            outputs that don't match the sinks or replace an input.
        BuilderFailure: `model_failed`, `invalid_model_output`, or
            `builder_internal_error` for anything else that went wrong.
    """
    _check_inputs(analysis, inputs)
    check_destinations(
        outputs, [s.options.output for s in analysis.sinks], inputs.values()
    )
    missing = sorted(set(analysis.models) - set(models))
    if missing:
        raise ValueError(f"no handle for the served models {missing}")
    try:
        return await _Run(analysis, models, concurrency, report_progress).run(
            inputs, outputs
        )
    except (InvalidInvocation, BuilderFailure):
        raise
    except Exception as error:
        logger.exception("analysis %s failed unexpectedly", analysis.id)
        raise BuilderFailure(
            code=FailureCode.INTERNAL_ERROR,
            safe_message=f"analysis {analysis.id!r} failed unexpectedly",
            retryable=True,
        ) from error


def _check_inputs(analysis: Analysis, inputs: Mapping[str, Path]) -> None:
    declared = analysis.inputs
    missing = sorted(
        n for n, required in declared.items() if required and n not in inputs
    )
    unknown = sorted(set(inputs) - set(declared))
    if missing or unknown:
        raise InvalidInvocation(
            f"inputs of {analysis.id}: {', '.join(sorted(declared))}; "
            f"missing required {missing or 'none'}, unknown {unknown or 'none'}"
        )


class _Run:
    """One run's state: steps bound to handles, timings, progress."""

    def __init__(
        self,
        analysis: Analysis,
        models: Mapping[str, ModelHandle],
        concurrency: Concurrency,
        report_progress: Callable[[float], None],
    ) -> None:
        self.analysis = analysis
        self.concurrency = concurrency
        self.report_progress = report_progress
        self.timings = Timings()
        self.steps = [self._bind(step, models) for step in analysis.steps]
        self.ordered = any(sink.ordered for sink in analysis.sinks)
        self.done = 0

    def _bind(
        self, step: Any, models: Mapping[str, ModelHandle]
    ) -> tuple[str, Callable[[Batch], Awaitable[Batch]]]:
        if isinstance(step, ModelStep):
            return f"model {step.model.id}", model_call(step, models[step.model.id])

        async def blocking(batch: Batch) -> Batch:
            return await asyncio.get_running_loop().run_in_executor(
                self.pool, step, batch
            )

        return step.name, blocking

    async def run(
        self, inputs: Mapping[str, Path], outputs: Mapping[str, Path]
    ) -> dict[str, Path]:
        analysis = self.analysis
        started = time.monotonic()
        with ExitStack() as stack:
            self.pool = stack.enter_context(
                ThreadPoolExecutor(max_workers=self.concurrency.read_threads)
            )
            scratch = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            staging = stack.enter_context(Staging(outputs))
            opened = stack.enter_context(
                analysis.source.open(dict(inputs), Run(scratch, analysis.last_model))
            )
            keys = list(opened.keys)
            sinks = _OpenSinks()
            try:
                for sink in analysis.sinks:
                    work_dir = staging.work_dir(sink.options.output)
                    sinks.open(
                        sink,
                        opened.frame,
                        Run(work_dir, analysis.last_model, analysis.columns),
                    )
                await self._process(keys, opened.load)
                artifacts = sinks.close(self.timings)
                written = staging.commit(artifacts)
            finally:
                sinks.abort()
                logger.info(
                    "analysis %s: %d of %d keys in %.1f s; per call: %s",
                    analysis.id,
                    self.done,
                    len(keys),
                    time.monotonic() - started,
                    self.timings,
                )
        if not keys:
            self.report_progress(1.0)
        return written

    async def _process(
        self, keys: Sequence[Any], load: Callable[[Sequence[Any]], Batch]
    ) -> None:
        loop = asyncio.get_running_loop()
        name = self.analysis.source.name
        total = len(keys)
        last_log = time.monotonic()

        def deliver(count: int, batch: Batch) -> None:
            nonlocal last_log
            if rows(batch):
                for sink in self.analysis.sinks:
                    self.timings.time(sink.name, sink.add, batch)
            self.done += count
            self.report_progress(self.done / total)
            if time.monotonic() - last_log >= PROGRESS_LOG_SECONDS:
                last_log = time.monotonic()
                logger.info(
                    "analysis %s: %d of %d keys; per call: %s",
                    self.analysis.id,
                    self.done,
                    total,
                    self.timings,
                )

        async def process(key: Any) -> tuple[int, Batch]:
            batch = await loop.run_in_executor(
                self.pool, self.timings.time, name, load, [key]
            )
            for label, step in self.steps:
                if not rows(batch):
                    break
                began = time.perf_counter()
                batch = await step(batch)
                self.timings.add(label, time.perf_counter() - began)
            if not self.ordered:
                deliver(1, batch)
            return 1, batch

        in_flight: deque[asyncio.Task[tuple[int, Batch]]] = deque()

        async def settle_one() -> None:
            if self.ordered:
                # The head stays registered until it is done, so cancelling
                # the run reaches it.
                result = await in_flight[0]
                in_flight.popleft()
                deliver(*result)
                return
            finished, _ = await asyncio.wait(
                in_flight, return_when=asyncio.FIRST_COMPLETED
            )
            for task in finished:
                in_flight.remove(task)
                task.result()

        try:
            for key in keys:
                if len(in_flight) >= self.concurrency.in_flight:
                    await settle_one()
                in_flight.append(asyncio.create_task(process(key)))
            while in_flight:
                await settle_one()
        finally:
            for task in in_flight:
                task.cancel()
            await asyncio.gather(*in_flight, return_exceptions=True)


class _OpenSinks:
    """The sinks opened so far; those not closed are aborted on failure."""

    def __init__(self) -> None:
        self._open: list[Sink] = []

    def open(self, sink: Sink, frame: dict[str, Any], run: Run) -> None:
        sink.open(frame, run)
        self._open.append(sink)

    def close(self, timings: Timings) -> dict[str, Path]:
        # A sink whose close fails cleans up after itself; the rest are aborted.
        artifacts = {}
        while self._open:
            sink = self._open.pop(0)
            artifacts[sink.options.output] = timings.time(sink.name, sink.close)
        return artifacts

    def abort(self) -> None:
        while self._open:
            sink = self._open.pop()
            try:
                sink.abort()
            except Exception:
                logger.exception("sink %s failed to abort", sink.name)
