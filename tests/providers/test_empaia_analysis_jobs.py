"""An EMPAIA job of an analysis's app: JES's route, admission, the runner, publication.

Each test posts a job to `/v1/empaia/analyses/{id}` as JES would, against a
fake App Service; the end-to-end one runs it on the analysis-runner core
with a fake model over `fake_slide()` and publishes the heatmap.
"""

from __future__ import annotations

import time
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import jwt
import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from model_service.analysis_runner.replica import AnalysisReplica
from model_service.analysis_runner.settings import ReplicaSettings
from model_service.api.scheduler.execution_scheduler import ExecutionScheduler
from model_service.api.scheduler.public import Priority
from model_service.api.scheduler.running_work import ActiveWorkRegistry
from model_service.api.scheduler.worker import ExecutionWorker
from model_service.catalog import load_catalog
from model_service.catalog.analyses import resolve_run
from model_service.providers.empaia.app_service.gateway import (
    EmpaiaInputState,
    EmpaiaJobContext,
    EmpaiaJobMode,
)
from model_service.providers.empaia.app_service.types import DirectAccess
from model_service.providers.empaia.jobs.adapter import EmpaiaProviderAdapter
from model_service.providers.empaia.jobs.inputs import EmpaiaInputAccessFactory
from model_service.providers.empaia.jobs.routes import create_router
from tests.components.helpers import fake_model, fake_slide
from tests.support.analysis_catalog import REGISTRY, served_model, write
from tests.support.empaia_fixtures import SLIDE_ID, StrictGateway, slide_info
from tests.support.memory_repository import InMemoryExecutionRepository


if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import numpy as np
    from rationai.empaia.typing import SlideInfo

    from model_service.catalog import Catalog
    from model_service.contracts.analysis_http import AnalysisRunRequest
    from model_service.contracts.json_values import JsonValue
    from model_service.contracts.result import RunResult
    from model_service.providers.empaia.app_service.gateway import MutationGuard


pytestmark = pytest.mark.anyio

API_URL = "https://empaia.invalid/app-api"
ANALYSIS = """
    source: {use: slide_tiles}
    steps: [{model: m1}]
    sinks: [{use: score_grid, output: heatmap}]
    expose:
      empaia:
        name: Test heatmap
        name_short: TestHM
        description: Finds things.
        modes:
          standalone: {parameters: [stride_fraction]}
          preprocessing:
        outputs:
          heatmap: {type: continuous_pixelmap}
"""


class AppService:
    """One EMPAIA job's App Service: what it answers and what it was told."""

    def __init__(self, slide: Path, mode: EmpaiaJobMode) -> None:
        self.slide = slide
        self.mode = mode
        self.configuration: dict[str, JsonValue] = {"aigopath_viewer": "{}"}
        self.inputs: dict[str, JsonValue] = {
            "stride_fraction": {"type": "float", "value": 1.0},
            "region_of_interest": {
                "upper_left": [0, 0],
                "width": 256,
                "height": 128,
            },
        }
        if mode is EmpaiaJobMode.PREPROCESSING:
            self.inputs = {}
        self.reads: list[str] = []
        self.progress: list[float] = []
        self.published: list[tuple[str, str]] = []
        self.finalized = False


class Gateway(StrictGateway):
    """The fake App Service through the gateway's SDK call families."""

    def __init__(self, service: AppService) -> None:
        self.service = service

    def set_before_mutation(self, callback: MutationGuard) -> None:
        self.guard = callback

    async def get_configuration(self) -> dict[str, JsonValue]:
        return dict(self.service.configuration)

    async def get_mode(self) -> EmpaiaJobMode:
        return self.service.mode

    async def has_input(self, name: str) -> bool:
        return name == "slide" or name in self.service.inputs

    async def get_input_data(self, name: str) -> JsonValue:
        self.service.reads.append(name)
        if name == "slide":
            return slide_info(256, 256, 4).model_dump(mode="json")
        return self.service.inputs[name]

    async def direct_access(self, name: str) -> DirectAccess | None:
        self.service.reads.append(f"{name} (direct access)")
        return DirectAccess(self.service.slide, supported=True)

    async def slide_info(self, name: str) -> SlideInfo:
        return slide_info(256, 256, 4)

    async def publish_pixelmap(
        self, name: str, slide_id: str, tile_sets: tuple[Any, ...]
    ) -> None:
        await self.guard()
        assert tile_sets
        self.service.published.append((name, slide_id))

    async def report_progress(self, fraction: float) -> None:
        self.service.progress.append(fraction)

    async def finalize(self) -> None:
        await self.guard()
        self.service.finalized = True

    async def close(self) -> None:
        return None


class Handle:
    """A served model's handle: the fake model, reporting a digest."""

    def __init__(self, digest: str) -> None:
        self.digest = digest
        self.model = fake_model(tile_size=64, mpp=0.5)

    async def describe(self) -> dict[str, Any]:
        return {"id": "m1", "digest": self.digest}

    async def infer(self, tile: np.ndarray) -> np.ndarray:
        return await self.model.infer(tile)


class Runner:
    """The analysis runner's dispatch: the replica core in this process."""

    def __init__(self, replica: AnalysisReplica) -> None:
        self.replica = replica

    async def run_analysis(self, request: AnalysisRunRequest) -> RunResult:
        return await self.replica.run(request)


class Deployment:
    """The JES route, scheduler, worker and runner, over one fake App Service."""

    def __init__(self, root: Path, catalog: Catalog, service: AppService) -> None:
        def gateway(context: EmpaiaJobContext) -> Gateway:
            assert context.api_url == API_URL
            return Gateway(service)

        workspace = root / "workspace"
        self.repository = InMemoryExecutionRepository()
        self.adapter = EmpaiaProviderAdapter(
            gateway,
            allowed_api_urls=(API_URL,),
            workspace_root=workspace,
            catalog=lambda: catalog,
        )
        digest = resolve_run(catalog, "a1").model_digests["m1"]
        runner = Runner(
            AnalysisReplica(
                ReplicaSettings(),
                models=lambda _id: Handle(digest),
                access={"empaia": EmpaiaInputAccessFactory(gateway, workspace)},
                progress_interval_seconds=0.001,
            )
        )
        self.scheduler = ExecutionScheduler(
            repository=self.repository,
            dispatch=runner,
            providers={"empaia": self.adapter},
            resolve_analysis=lambda i, s: resolve_run(catalog, i, s),
            default_timeout=timedelta(hours=1),
            maximum_timeout=timedelta(hours=6),
        )
        self.worker = ExecutionWorker(
            repository=self.repository,
            active_work=ActiveWorkRegistry(),
            providers={"empaia": self.adapter},
            dispatch=runner,
            owner="worker-1",
            lease_duration=timedelta(minutes=1),
        )

    async def client(self) -> AsyncIterator[AsyncClient]:
        app = FastAPI()
        app.include_router(create_router(self.scheduler))
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://model-service"
        ) as client:
            yield client


def job(job_id: str = "job-1") -> dict[str, str]:
    """What JES posts: the job's App API, ID and token."""
    token = jwt.encode(
        {"exp": time.time() + 3600}, "a-key-nobody-checks-in-this-test-1234"
    )
    return {"app_api": API_URL, "job_id": job_id, "access_token": token}


@pytest.fixture
def catalog(tmp_path: Path) -> Catalog:
    root = tmp_path / "catalog"
    write(root, "served-models/m1.yaml", served_model())
    write(root, "analyses/a1.yaml", ANALYSIS)
    write(
        root,
        "analyses/plain.yaml",
        ANALYSIS[: ANALYSIS.index("    expose:")],
    )
    return load_catalog(root, registry=REGISTRY)


@pytest.fixture
def slide(tmp_path: Path) -> Path:
    (tmp_path / "slides").mkdir()
    return fake_slide(tmp_path / "slides" / "slide.tiff")


async def test_a_standalone_job_runs_the_analysis_and_publishes_its_heatmap(
    tmp_path: Path, catalog: Catalog, slide: Path
) -> None:
    service = AppService(slide, EmpaiaJobMode.STANDALONE)
    deployment = Deployment(tmp_path, catalog, service)
    async for client in deployment.client():
        posted = await client.post("/v1/empaia/analyses/a1", json=job())
        execution_id = posted.headers["location"].rsplit("/", 1)[1]
        stored = await deployment.repository.get(execution_id)

        assert await deployment.worker.run_once()
        assert await deployment.worker.run_once()
        status = (await client.get(f"/v1/empaia/executions/{execution_id}")).json()

    assert posted.status_code == 202
    assert status == {"status": "succeeded", "message": None}
    # Admission: the analysis, resolved with the user's parameter, and typed inputs.
    assert stored.target.analysis_id == "a1"
    assert stored.target.analysis is not None
    assert stored.target.analysis["source"]["stride_fraction"] == 1.0
    assert stored.priority == Priority.HIGH
    state = EmpaiaInputState.model_validate(stored.invocation.input_access.state)
    assert state.inputs == {
        "slide": {"type": "wsi"},
        "region_of_interest": {"type": "rectangle"},
    }
    # The run: the slide linked, the rectangle made a mask, progress reported.
    assert service.reads == [
        "stride_fraction",
        "slide (direct access)",
        "region_of_interest",
        "slide",
    ]
    assert service.progress
    # Publication: the heatmap on the job's slide, then finalization.
    assert service.published == [("heatmap", SLIDE_ID)]
    assert service.finalized
    assert not (tmp_path / "workspace").exists() or not any(
        (tmp_path / "workspace").iterdir()
    )


async def test_a_preprocessing_job_gets_the_slide_alone(
    tmp_path: Path, catalog: Catalog, slide: Path
) -> None:
    service = AppService(slide, EmpaiaJobMode.PREPROCESSING)
    deployment = Deployment(tmp_path, catalog, service)
    async for client in deployment.client():
        posted = await client.post("/v1/empaia/analyses/a1", json=job())
        execution_id = posted.headers["location"].rsplit("/", 1)[1]
        stored = await deployment.repository.get(execution_id)

    assert list(stored.invocation.inputs) == ["slide"]
    # No parameters: EMPAIA starts preprocessing alone, on the file's settings.
    assert stored.target.analysis is not None
    assert stored.target.analysis["source"]["stride_fraction"] == 0.5
    assert stored.priority == Priority.NORMAL
    assert service.reads == []


async def test_the_same_job_posted_twice_runs_once(
    tmp_path: Path, catalog: Catalog, slide: Path
) -> None:
    deployment = Deployment(
        tmp_path, catalog, AppService(slide, EmpaiaJobMode.PREPROCESSING)
    )
    async for client in deployment.client():
        first = await client.post("/v1/empaia/analyses/a1", json=job())
        again = await client.post("/v1/empaia/analyses/a1", json=job())

    assert (first.status_code, again.status_code) == (202, 200)
    assert first.headers["location"] == again.headers["location"]


@pytest.mark.parametrize(
    ("analysis_id", "status", "code"),
    [
        ("plain", 422, "invalid_request"),
        ("missing", 404, "unknown_analysis"),
    ],
)
async def test_an_analysis_not_offered_in_empaia_is_refused(
    tmp_path: Path,
    catalog: Catalog,
    slide: Path,
    analysis_id: str,
    status: int,
    code: str,
) -> None:
    deployment = Deployment(
        tmp_path, catalog, AppService(slide, EmpaiaJobMode.STANDALONE)
    )
    async for client in deployment.client():
        refused = await client.post(f"/v1/empaia/analyses/{analysis_id}", json=job())

    assert refused.status_code == status
    assert refused.json()["detail"]["code"] == code
