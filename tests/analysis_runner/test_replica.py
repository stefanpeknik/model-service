"""The analysis-runner replica core: a run request in, outputs on shared storage out.

Each test resolves a small analysis over `fake_slide()` the way the API does,
and runs it with fake served-model handles below a shared-storage root.
"""

from __future__ import annotations

import asyncio
import copy
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest

from model_service.analysis_runner.replica import AnalysisReplica
from model_service.analysis_runner.settings import ReplicaSettings
from model_service.catalog import load_catalog
from model_service.catalog.analyses import resolve_run
from model_service.contracts.analysis_http import AnalysisRunRequest
from model_service.contracts.failures import BuilderFailure, InvalidInvocation
from model_service.contracts.invocation import (
    InputAccessContext,
    Invocation,
    SharedPathInput,
)
from model_service.contracts.result import BuilderResult, SharedArtifact
from model_service.formats import score_grid
from model_service.testing import fake_model, fake_slide
from tests.support.analysis_catalog import REGISTRY, served_model, write


if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np

    from model_service.catalog.analyses import ResolvedAnalysis


class Handle:
    """A served model's handle: the fake model, reporting a digest."""

    def __init__(self, digest: str) -> None:
        self.digest = digest
        self.model = fake_model(tile_size=64, mpp=0.5)
        self.calls = 0

    async def describe(self) -> dict[str, Any]:
        return {"id": "m1", "digest": self.digest}

    async def infer(self, tile: np.ndarray) -> np.ndarray:
        self.calls += 1
        return await self.model.infer(tile)


@pytest.fixture
def resolved(tmp_path: Path) -> ResolvedAnalysis:
    root = tmp_path / "catalog"
    write(root, "served-models/m1.yaml", served_model())
    write(
        root,
        "analyses/a1.yaml",
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
    resolved: ResolvedAnalysis,
    root: Path,
    *,
    output_path: str | None = "out/heatmap.npz",
    deadline: timedelta = timedelta(minutes=5),
    **inputs: str,
) -> AnalysisRunRequest:
    return AnalysisRunRequest(
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
        analysis=resolved.document,
    )


def run(
    run_request: AnalysisRunRequest,
    handle: Handle,
    allowed: Path,
) -> BuilderResult:
    replica = AnalysisReplica(
        ReplicaSettings(shared_storage_roots=(allowed,)),
        models=lambda model_id: handle,
    )
    return asyncio.run(replica.run(run_request))


def test_a_run_writes_its_output_where_the_request_says(
    resolved: ResolvedAnalysis, storage: Path
) -> None:
    handle = Handle(resolved.model_digests["m1"])

    result = run(request(resolved, storage), handle, storage)

    path = storage / "out" / "heatmap.npz"
    assert result == BuilderResult(
        {"heatmap": SharedArtifact(str(path), score_grid.MEDIA_TYPE)}
    )
    assert score_grid.check(path) == []
    assert score_grid.read(path).values.shape == (1, 8, 8)
    assert handle.calls == 49


def test_without_an_output_path_the_output_is_named_after_itself(
    resolved: ResolvedAnalysis, storage: Path
) -> None:
    handle = Handle(resolved.model_digests["m1"])

    result = run(request(resolved, storage, output_path=None), handle, storage)

    assert result.outputs["heatmap"].path == str(storage / "heatmap.npz")  # type: ignore[union-attr]


def test_a_served_model_running_another_declaration_is_a_contract_mismatch(
    resolved: ResolvedAnalysis, storage: Path
) -> None:
    handle = Handle("0" * 64)

    with pytest.raises(BuilderFailure) as raised:
        run(request(resolved, storage), handle, storage)

    assert (raised.value.code, raised.value.retryable) == (
        "model_contract_mismatch",
        False,
    )
    assert "m1" in raised.value.safe_message
    assert handle.calls == 0
    assert not (storage / "out").exists()


def test_a_root_outside_the_allowed_ones_is_refused(
    resolved: ResolvedAnalysis, storage: Path, tmp_path: Path
) -> None:
    other = tmp_path / "elsewhere"
    other.mkdir()
    handle = Handle(resolved.model_digests["m1"])

    with pytest.raises(InvalidInvocation, match="not allowed"):
        run(request(resolved, storage), handle, other)
    with pytest.raises(InvalidInvocation):
        run(request(resolved, storage, output_path="../escape.npz"), handle, storage)
    assert handle.calls == 0


def test_inputs_the_analysis_does_not_read_are_refused(
    resolved: ResolvedAnalysis, storage: Path
) -> None:
    handle = Handle(resolved.model_digests["m1"])

    with pytest.raises(InvalidInvocation, match="mask"):
        run(request(resolved, storage, mask="slide.tiff"), handle, storage)


def test_a_document_changed_after_admission_is_refused(
    resolved: ResolvedAnalysis, storage: Path
) -> None:
    changed = copy.deepcopy(resolved.document)
    changed["source"]["stride_fraction"] = 0.25
    run_request = request(resolved, storage)
    run_request = AnalysisRunRequest(run_request.invocation, changed)

    with pytest.raises(InvalidInvocation, match="digest"):
        run(run_request, Handle(resolved.model_digests["m1"]), storage)


def test_a_run_past_its_deadline_fails_without_output(
    resolved: ResolvedAnalysis, storage: Path
) -> None:
    run_request = request(resolved, storage, deadline=timedelta(seconds=-1))

    with pytest.raises(BuilderFailure) as raised:
        run(run_request, Handle(resolved.model_digests["m1"]), storage)

    assert raised.value.code == "builder_deadline_exceeded"
    assert [p for p in storage.rglob("*") if p.is_file()] == [storage / "slide.tiff"]
