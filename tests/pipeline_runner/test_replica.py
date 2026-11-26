"""The pipeline-runner replica core: a run request in, outputs on shared storage out.

Each test resolves a small pipeline over `fake_slide()` the way the control plane does,
and runs it with fake model handles below a shared-storage root.
"""

from __future__ import annotations

import asyncio
import copy
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest

from model_service.catalog import load_catalog
from model_service.catalog.pipelines import resolve_run
from model_service.contracts.failures import InvalidInvocation, RunFailure
from model_service.contracts.invocation import (
    ExecutionContext,
    InputAccessContext,
    Invocation,
    SharedPathInput,
)
from model_service.contracts.pipeline_http import PipelineRunRequest
from model_service.contracts.result import ModelRun, RunResult, SharedArtifact
from model_service.formats import score_grid
from model_service.pipeline_runner.replica import PipelineReplica
from model_service.pipeline_runner.settings import ReplicaSettings
from model_service.providers.input_access import InputAccess, InputAccessFactory
from model_service.providers.standalone.shared_storage import SharedStorageAccessFactory
from tests.components.helpers import fake_model, fake_slide
from tests.support.pipeline_catalog import REGISTRY, model_file, write


if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from pathlib import Path

    import numpy as np

    from model_service.catalog.pipelines import ResolvedPipeline


class Handle:
    """A model's handle: the fake model, reporting a digest."""

    def __init__(
        self, digest: str, weights_revision: str = "/weights/model.onnx"
    ) -> None:
        self.digest = digest
        self.weights_revision = weights_revision
        self.model = fake_model(tile_size=64, mpp=0.5)
        self.calls = 0

    async def describe(self) -> dict[str, Any]:
        return {
            "id": "m1",
            "digest": self.digest,
            "weights_revision": self.weights_revision,
            "execution_providers": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        }

    async def infer(self, tile: np.ndarray) -> np.ndarray:
        self.calls += 1
        return await self.model.infer(tile)


@pytest.fixture
def resolved(tmp_path: Path) -> ResolvedPipeline:
    root = tmp_path / "catalog"
    write(root, "models/m1.yaml", model_file())
    write(
        root,
        "pipelines/a1.yaml",
        """
        source: {use: slide_tiles, skip_background: false}
        steps: [{model: m1}]
        sinks: [{use: score_grid, output: heatmap}]
        """,
    )
    return resolve_run(load_catalog(root, registry=REGISTRY), "a1")


@pytest.fixture
def storage(tmp_path: Path) -> Path:
    root = tmp_path / "shared"
    root.mkdir()
    fake_slide(root / "slide.tiff")
    return root


def request(
    resolved: ResolvedPipeline,
    root: Path,
    *,
    output_path: str | None = "out/heatmap.npz",
    deadline: timedelta = timedelta(minutes=5),
    **inputs: str,
) -> PipelineRunRequest:
    return PipelineRunRequest(
        invocation=Invocation(
            inputs={
                name: SharedPathInput(path)
                for name, path in ({"slide": "slide.tiff"} | inputs).items()
            },
            configuration=None,
            input_access=InputAccessContext("shared_storage", {"root": str(root)}),
            deadline_at=datetime.now(UTC) + deadline,
            output_path=output_path,
        ),
        pipeline=resolved.document,
    )


def run(
    run_request: PipelineRunRequest,
    handle: Handle,
    allowed: Path,
) -> RunResult:
    replica = PipelineReplica(
        ReplicaSettings(shared_storage_roots=(allowed,)),
        models=lambda model_id: handle,
    )
    return asyncio.run(replica.run(run_request))


def test_a_run_writes_its_output_where_the_request_says(
    resolved: ResolvedPipeline, storage: Path
) -> None:
    handle = Handle(resolved.model_digests["m1"])

    result = run(request(resolved, storage), handle, storage)

    path = storage / "out" / "heatmap.npz"
    assert result == RunResult(
        {"heatmap": SharedArtifact(str(path), score_grid.FORMAT.media_type)},
        models={
            "m1": ModelRun(
                digest=resolved.model_digests["m1"],
                weights_revision="/weights/model.onnx",
                execution_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        },
    )
    assert score_grid.read(path).values.shape == (1, 8, 8)
    assert handle.calls == 49


def test_without_an_output_path_the_output_is_named_after_itself(
    resolved: ResolvedPipeline, storage: Path
) -> None:
    handle = Handle(resolved.model_digests["m1"])

    result = run(request(resolved, storage, output_path=None), handle, storage)

    assert result.outputs["heatmap"].path == str(storage / "heatmap.npz")  # type: ignore[union-attr]


def test_a_model_running_another_declaration_is_a_retryable_contract_mismatch(
    resolved: ResolvedPipeline, storage: Path
) -> None:
    handle = Handle("0" * 64)

    with pytest.raises(RunFailure) as raised:
        run(request(resolved, storage), handle, storage)

    assert (raised.value.code, raised.value.retryable) == (
        "model_contract_mismatch",
        True,
    )
    assert "m1" in raised.value.safe_message
    assert handle.calls == 0
    assert not (storage / "out").exists()


def test_a_model_that_does_not_say_what_it_runs_is_a_contract_mismatch(
    resolved: ResolvedPipeline, storage: Path
) -> None:
    class Silent(Handle):
        async def describe(self) -> dict[str, Any]:
            return {"id": "m1", "digest": self.digest}

    handle = Silent(resolved.model_digests["m1"])

    with pytest.raises(RunFailure) as raised:
        run(request(resolved, storage), handle, storage)

    assert (raised.value.code, raised.value.retryable) == (
        "model_contract_mismatch",
        True,
    )
    assert handle.calls == 0


URI = "file:///weights/model.onnx"
SHA256 = "a" * 64


@pytest.mark.parametrize(
    ("running", "refused"),
    [(f"{URI}@sha256:{SHA256}", False), (f"{URI}@sha256:{'b' * 64}", True)],
)
def test_a_model_must_run_the_weights_its_file_pins(
    tmp_path: Path, storage: Path, running: str, refused: bool
) -> None:
    root = tmp_path / "catalog"
    pinned = model_file().replace(
        "{use: local, path: /weights/model.onnx}",
        f'{{use: mlflow, uri: "{URI}", sha256: "{SHA256}"}}',
    )
    write(root, "models/m1.yaml", pinned)
    write(
        root,
        "pipelines/a1.yaml",
        """
        source: {use: slide_tiles, skip_background: false}
        steps: [{model: m1}]
        sinks: [{use: score_grid, output: heatmap}]
        """,
    )
    resolved = resolve_run(load_catalog(root, registry=REGISTRY), "a1")
    handle = Handle(resolved.model_digests["m1"], running)

    if not refused:
        run(request(resolved, storage), handle, storage)
        assert handle.calls > 0
        return
    with pytest.raises(RunFailure) as raised:
        run(request(resolved, storage), handle, storage)
    assert (raised.value.code, raised.value.retryable) == (
        "model_contract_mismatch",
        True,
    )
    assert "weights" in raised.value.safe_message
    assert handle.calls == 0


def test_a_root_outside_the_allowed_ones_is_refused(
    resolved: ResolvedPipeline, storage: Path, tmp_path: Path
) -> None:
    other = tmp_path / "elsewhere"
    other.mkdir()
    handle = Handle(resolved.model_digests["m1"])

    with pytest.raises(InvalidInvocation, match="not allowed"):
        run(request(resolved, storage), handle, other)
    with pytest.raises(InvalidInvocation):
        run(request(resolved, storage, output_path="../escape.npz"), handle, storage)
    assert handle.calls == 0


def test_inputs_the_pipeline_does_not_read_are_refused(
    resolved: ResolvedPipeline, storage: Path
) -> None:
    handle = Handle(resolved.model_digests["m1"])

    with pytest.raises(InvalidInvocation, match="mask"):
        run(request(resolved, storage, mask="slide.tiff"), handle, storage)


def test_a_document_changed_after_admission_is_refused(
    resolved: ResolvedPipeline, storage: Path
) -> None:
    changed = copy.deepcopy(resolved.document)
    changed["source"]["stride_fraction"] = 0.25
    run_request = request(resolved, storage)
    run_request = PipelineRunRequest(run_request.invocation, changed)

    with pytest.raises(InvalidInvocation, match="digest"):
        run(run_request, Handle(resolved.model_digests["m1"]), storage)


def test_a_run_past_its_deadline_fails_without_output(
    resolved: ResolvedPipeline, storage: Path
) -> None:
    run_request = request(resolved, storage, deadline=timedelta(seconds=-1))

    with pytest.raises(RunFailure) as raised:
        run(run_request, Handle(resolved.model_digests["m1"]), storage)

    assert raised.value.code == "run_deadline_exceeded"
    assert [p for p in storage.rglob("*") if p.is_file()] == [storage / "slide.tiff"]


class ClosingAccess(InputAccess):
    """Shared-storage access whose close the test decides."""

    def __init__(
        self, shared: InputAccess, close: Callable[[], Awaitable[None]]
    ) -> None:
        self._shared = shared
        self._close = close

    def input_path(self, name: str) -> Path:
        return self._shared.input_path(name)

    def output_path(self, relative_path: str) -> Path:
        return self._shared.output_path(relative_path)

    async def aclose(self) -> None:
        await self._close()


class Access(InputAccessFactory):
    """Shared-storage access factory whose accesses close as the test decides."""

    def __init__(self, storage: Path, close: Callable[[], Awaitable[None]]) -> None:
        self._storage = storage
        self._close = close

    async def create(
        self, invocation: Invocation, execution_context: ExecutionContext
    ) -> InputAccess:
        shared = await SharedStorageAccessFactory([self._storage]).create(
            invocation, execution_context
        )
        return ClosingAccess(shared, self._close)


def replica(storage: Path, access: Any, **options: Any) -> PipelineReplica:
    return PipelineReplica(
        ReplicaSettings(shared_storage_roots=(storage,)),
        access={"shared_storage": access},
        **options,
    )


async def failing_close() -> None:
    raise OSError("provider credential is secret")


def test_a_failed_cleanup_keeps_the_runs_outcome(
    resolved: ResolvedPipeline, storage: Path
) -> None:
    handle = Handle(resolved.model_digests["m1"])
    core = replica(storage, Access(storage, failing_close), models=lambda _: handle)

    result = asyncio.run(core.run(request(resolved, storage)))
    with pytest.raises(RunFailure) as raised:
        handle.digest = "0" * 64
        asyncio.run(core.run(request(resolved, storage)))

    assert set(result.outputs) == {"heatmap"}
    assert raised.value.code == "model_contract_mismatch"


def test_the_deadline_covers_opening_the_inputs(
    resolved: ResolvedPipeline, storage: Path
) -> None:
    cancelled = asyncio.Event()

    class Blocking:
        async def create(self, *_args: object) -> InputAccess:
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()
            raise AssertionError

    handle = Handle(resolved.model_digests["m1"])
    run_request = request(resolved, storage, deadline=timedelta(milliseconds=50))

    with pytest.raises(RunFailure) as raised:
        asyncio.run(
            asyncio.wait_for(
                replica(storage, Blocking(), models=lambda _: handle).run(run_request),
                timeout=1,
            )
        )

    assert (raised.value.code, raised.value.retryable) == (
        "run_deadline_exceeded",
        False,
    )
    assert cancelled.is_set()


def test_cleanup_is_bounded(resolved: ResolvedPipeline, storage: Path) -> None:
    async def scenario() -> RunResult:
        close_cancelled = asyncio.Event()

        async def hanging_close() -> None:
            try:
                await asyncio.Event().wait()
            finally:
                close_cancelled.set()

        handle = Handle(resolved.model_digests["m1"])
        core = replica(
            storage,
            Access(storage, hanging_close),
            models=lambda _: handle,
            cleanup_timeout_seconds=0.02,
        )
        result = await asyncio.wait_for(core.run(request(resolved, storage)), 5)
        await asyncio.wait_for(close_cancelled.wait(), 1)
        return result

    assert set(asyncio.run(scenario()).outputs) == {"heatmap"}


def test_a_cancelled_run_waits_for_cleanup_however_often_cancelled(
    resolved: ResolvedPipeline, storage: Path
) -> None:
    async def scenario() -> None:
        started = asyncio.Event()
        closing = asyncio.Event()
        release = asyncio.Event()
        closed = asyncio.Event()

        class Stuck(Handle):
            async def infer(self, tile: np.ndarray) -> np.ndarray:
                started.set()
                await asyncio.Event().wait()
                raise AssertionError

        async def slow_close() -> None:
            closing.set()
            await release.wait()
            closed.set()

        handle = Stuck(resolved.model_digests["m1"])
        core = replica(storage, Access(storage, slow_close), models=lambda _: handle)
        task = asyncio.create_task(core.run(request(resolved, storage)))
        await asyncio.wait_for(started.wait(), 5)
        task.cancel()
        await asyncio.wait_for(closing.wait(), 1)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        assert closed.is_set()

    asyncio.run(scenario())
