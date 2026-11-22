"""The analysis routes: the catalog listing and synchronous runs, with a fake runner."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from httpx import ASGITransport, AsyncClient

from model_service.api.app_factory import create_app
from model_service.api.model_catalog import ModelCatalog, served_model_binding
from model_service.api.scheduler.builder_dispatch import (
    ConfirmedRejectionError,
    RemoteFailureError,
)
from model_service.api.synchronous_runs import SynchronousAnalysisService
from model_service.catalog import load_catalog
from model_service.catalog.analyses import load_resolved, resolve_run
from model_service.contracts.result import BuilderResult, SharedArtifact
from model_service.formats import score_grid


if TYPE_CHECKING:
    from fastapi import FastAPI

    from model_service.contracts.analysis_http import AnalysisRunRequest


pytestmark = pytest.mark.anyio

CATALOG = load_catalog(Path(__file__).resolve().parents[2] / "catalog", "rationai-test")
RUN = "/v1/analyses/prostate-heatmap/run"
RESULT = BuilderResult(
    {"heatmap": SharedArtifact("/mnt/shared/out.npz", score_grid.MEDIA_TYPE)}
)


class FakeRunner:
    """Records the runs sent to the analysis runner."""

    def __init__(self, error: Exception | None = None) -> None:
        self.requests: list[AnalysisRunRequest] = []
        self.error = error

    async def run_analysis(self, request: AnalysisRunRequest) -> BuilderResult:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return RESULT


class NoScheduler:
    alive = True


def app(runner: FakeRunner | None) -> FastAPI:
    return create_app(
        NoScheduler(),  # type: ignore[arg-type]
        catalog=ModelCatalog([], resolved=CATALOG),
        analysis_runner=(
            None
            if runner is None
            else SynchronousAnalysisService(lambda: CATALOG, runner)
        ),
    )


async def post(
    runner: FakeRunner | None, path: str = RUN, **changes: object
) -> tuple[int, dict[str, object]]:
    body: dict[str, object] = {
        "inputs": {"slide": {"kind": "shared_path", "path": "slides/a.mrxs"}},
        "input_access": {"kind": "shared_storage", "state": {"root": "/mnt/shared"}},
        "output_path": "out.npz",
        "deadline_at": (datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
        **changes,
    }
    async with AsyncClient(
        transport=ASGITransport(app=app(runner)), base_url="http://model-service"
    ) as client:
        response = await client.post(path, json=body)
    return response.status_code, response.json()


async def test_analyses_list_inputs_outputs_settings_and_models() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app(FakeRunner())), base_url="http://model-service"
    ) as client:
        response = await client.get("/v1/analyses")

    assert response.status_code == 200
    (listed,) = [a for a in response.json() if a["id"] == "prostate-heatmap"]
    schema = listed.pop("settings_schema")
    assert listed == {
        "id": "prostate-heatmap",
        "summary": "Scores each tile of a prostate slide into a heatmap.",
        "inputs": {
            "slide": {"required": True},
            "region_of_interest": {"required": False},
        },
        "outputs": {
            "heatmap": {
                "kind": "shared_artifact",
                "format": "score_grid",
                "media_type": "application/vnd.rationai.score-grid+npz; version=1",
            }
        },
        "settings": {
            "min_tissue_fraction": 0.0,
            "skip_background": True,
            "stride_fraction": 0.5,
        },
        "models": ["prostate-classifier-1"],
        "exposure": {},
    }
    assert schema["additionalProperties"] is False
    assert {k: v["default"] for k, v in schema["properties"].items()} == {
        "min_tissue_fraction": 0.0,
        "skip_background": True,
        "stride_fraction": 0.5,
    }


async def test_the_embeddings_analyses_list_an_embedding_table() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app(FakeRunner())), base_url="http://model-service"
    ) as client:
        response = await client.get("/v1/analyses")

    listed = {a["id"]: a for a in response.json()}
    for analysis, model in (
        ("virchow2-embeddings", "virchow2"),
        ("prov-gigapath-embeddings", "prov-gigapath"),
    ):
        assert listed[analysis]["models"] == [model]
        assert listed[analysis]["outputs"] == {
            "embeddings": {
                "kind": "shared_artifact",
                "format": "embedding_table",
                "media_type": "application/vnd.rationai.embedding-table+hdf5; "
                "version=2",
            }
        }
        assert listed[analysis]["settings"]["stride_fraction"] == 1.0


async def test_a_run_sends_the_analysis_resolved_with_its_settings() -> None:
    runner = FakeRunner()

    status, body = await post(runner, configuration={"stride_fraction": 1.0})

    assert status == 200, body
    assert body == {
        "outputs": {
            "heatmap": {
                "kind": "shared_artifact",
                "path": "/mnt/shared/out.npz",
                "media_type": score_grid.MEDIA_TYPE,
            }
        }
    }
    (sent,) = runner.requests
    expected = resolve_run(CATALOG, "prostate-heatmap", {"stride_fraction": 1.0})
    assert sent.analysis == expected.document
    assert load_resolved(sent.analysis).analysis.source.options.stride_fraction == 1.0  # type: ignore[attr-defined]
    assert sent.invocation.output_path == "out.npz"
    assert sent.invocation.configuration == {
        "min_tissue_fraction": 0.0,
        "skip_background": True,
        "stride_fraction": 1.0,
    }


@pytest.mark.parametrize(
    ("path", "changes", "status", "code"),
    [
        ("/v1/analyses/missing/run", {}, 404, "unknown_analysis"),
        (RUN, {"configuration": {"model_id": "x"}}, 422, "invalid_request"),
        (RUN, {"configuration": {"stride_fraction": -1}}, 422, "invalid_request"),
        (RUN, {"inputs": {}}, 422, "invalid_request"),
        (
            RUN,
            {
                "inputs": {
                    "slide": {"kind": "shared_path", "path": "a.mrxs"},
                    "mask": {"kind": "shared_path", "path": "m.tiff"},
                }
            },
            422,
            "invalid_request",
        ),
        (RUN, {"deadline_at": "2000-01-01T00:00:00+00:00"}, 422, "invalid_request"),
    ],
)
async def test_a_run_the_analysis_cannot_take_is_refused_before_dispatch(
    path: str, changes: dict[str, object], status: int, code: str
) -> None:
    runner = FakeRunner()

    answered, body = await post(runner, path, **changes)

    assert (answered, body["detail"]["code"]) == (status, code)  # type: ignore[index]
    assert runner.requests == []


async def test_an_unknown_setting_is_named() -> None:
    _, body = await post(FakeRunner(), configuration={"model_id": "x"})

    assert "model_id" in body["detail"]["message"]  # type: ignore[index]


@pytest.mark.parametrize(
    ("error", "status", "code", "retryable"),
    [
        (
            RemoteFailureError(
                code="model_contract_mismatch",
                safe_message="model changed",
                retryable=False,
            ),
            422,
            "model_contract_mismatch",
            False,
        ),
        (
            ConfirmedRejectionError(timedelta(seconds=7)),
            429,
            "builder_backpressure",
            True,
        ),
    ],
)
async def test_the_runners_failures_reach_the_caller_safely(
    error: Exception, status: int, code: str, retryable: bool
) -> None:
    answered, body = await post(FakeRunner(error))

    assert (answered, body["detail"]) == (
        status,
        {"code": code, "message": body["detail"]["message"], "retryable": retryable},  # type: ignore[index]
    )


async def test_runs_are_unavailable_without_a_runner() -> None:
    status, body = await post(None)

    assert (status, body["detail"]["code"]) == (503, "builder_unavailable")  # type: ignore[index]


async def test_models_list_served_models_with_declarations_and_analyses() -> None:
    served = CATALOG.entries("served-models")
    catalog = ModelCatalog(
        [served_model_binding(entry.value) for entry in served.values()],
        resolved=CATALOG,
    )
    app = create_app(NoScheduler(), catalog=catalog)  # type: ignore[arg-type]
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://model-service"
    ) as client:
        listed = (await client.get("/v1/models")).json()
        single = (await client.get("/v1/models/prostate-classifier-1")).json()

    by_id = {model["id"]: model for model in listed}
    assert sorted(by_id) == [
        "episeg-1",
        "prostate-classifier-1",
        "prov-gigapath",
        "virchow2",
    ]
    prostate = by_id["prostate-classifier-1"]
    assert single == prostate
    assert prostate["analyses"] == ["prostate-heatmap"]
    assert prostate["declaration"]["input"] == {"tile_size": 512, "mpp": 0.468}
    # The environment's patch is part of the declaration.
    assert prostate["declaration"]["weights"]["uri"].startswith(
        "mlflow-artifacts:/169/"
    )
    assert prostate["digest"] == served["prostate-classifier-1"].value.digest
    assert by_id["virchow2"]["analyses"] == ["virchow2-embeddings"]
