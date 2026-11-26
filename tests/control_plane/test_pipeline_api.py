"""The pipeline routes: the catalog listing and synchronous runs, with a fake runner."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import pytest
from httpx import ASGITransport, AsyncClient

from model_service.catalog import load_catalog
from model_service.catalog.pipelines import load_resolved, resolve_run
from model_service.contracts.result import ModelRun, RunResult, SharedArtifact
from model_service.control_plane.app_factory import create_app, pipeline_resolver
from model_service.control_plane.scheduler.dispatch import (
    ConfirmedRejectionError,
    RemoteFailureError,
    UnknownOutcomeError,
)
from model_service.control_plane.synchronous_runs import SynchronousPipelineService
from model_service.formats import score_grid
from tests.support.paths import CATALOG as CATALOG_DIR
from tests.support.paths import RATIONAI_TEST


if TYPE_CHECKING:
    from fastapi import FastAPI

    from model_service.contracts.pipeline_http import PipelineRunRequest


pytestmark = pytest.mark.anyio

CATALOG = load_catalog(CATALOG_DIR, RATIONAI_TEST)
RUN = "/v1/pipelines/prostate-heatmap/run"
RESULT = RunResult(
    {"heatmap": SharedArtifact("/mnt/shared/out.npz", score_grid.FORMAT.media_type)},
    models={
        "prostate-classifier-1": ModelRun(
            digest="d" * 64,
            weights_revision="/weights/prostate.onnx",
            execution_providers=["CPUExecutionProvider"],
        )
    },
)


class FakeRunner:
    """Records the runs sent to the pipeline runner."""

    def __init__(self, error: Exception | None = None) -> None:
        self.requests: list[PipelineRunRequest] = []
        self.error = error

    async def run_pipeline(self, request: PipelineRunRequest) -> RunResult:
        self.requests.append(request)
        if self.error is not None:
            raise self.error
        return RESULT


class NoScheduler:
    alive = True


def app(runner: FakeRunner | None) -> FastAPI:
    return create_app(
        NoScheduler(),  # type: ignore[arg-type]
        catalog=lambda: CATALOG,
        pipeline_runner=(
            None
            if runner is None
            else SynchronousPipelineService(pipeline_resolver(lambda: CATALOG), runner)
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


async def test_pipelines_list_inputs_outputs_settings_and_models() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app(FakeRunner())), base_url="http://model-service"
    ) as client:
        response = await client.get("/v1/pipelines")

    assert response.status_code == 200
    (listed,) = [a for a in response.json() if a["id"] == "prostate-heatmap"]
    schema = listed.pop("settings_schema")
    providers = listed.pop("providers")
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
    }
    assert providers["empaia"]["name"] == "Prostate cancer heatmap"
    assert providers["empaia"]["outputs"] == {
        "heatmap": {"type": "continuous_pixelmap"}
    }
    assert schema["additionalProperties"] is False
    assert {k: v["default"] for k, v in schema["properties"].items()} == {
        "min_tissue_fraction": 0.0,
        "skip_background": True,
        "stride_fraction": 0.5,
    }


async def test_the_embeddings_pipelines_list_an_embedding_table() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app(FakeRunner())), base_url="http://model-service"
    ) as client:
        response = await client.get("/v1/pipelines")

    listed = {a["id"]: a for a in response.json()}
    for pipeline, model in (
        ("virchow2-embeddings", "virchow2"),
        ("prov-gigapath-embeddings", "prov-gigapath"),
    ):
        assert listed[pipeline]["models"] == [model]
        assert listed[pipeline]["outputs"] == {
            "embeddings": {
                "kind": "shared_artifact",
                "format": "embedding_table",
                "media_type": "application/vnd.rationai.embedding-table+hdf5; "
                "version=2",
            }
        }
        assert listed[pipeline]["settings"]["stride_fraction"] == 1.0


async def test_a_run_sends_the_pipeline_resolved_with_its_settings() -> None:
    runner = FakeRunner()

    status, body = await post(runner, configuration={"stride_fraction": 1.0})

    assert status == 200, body
    assert body == {
        "outputs": {
            "heatmap": {
                "kind": "shared_artifact",
                "path": "/mnt/shared/out.npz",
                "media_type": score_grid.FORMAT.media_type,
            }
        },
        "models": {
            "prostate-classifier-1": {
                "digest": "d" * 64,
                "weights_revision": "/weights/prostate.onnx",
                "execution_providers": ["CPUExecutionProvider"],
            }
        },
    }
    (sent,) = runner.requests
    expected = resolve_run(CATALOG, "prostate-heatmap", {"stride_fraction": 1.0})
    assert sent.pipeline == expected.document
    assert load_resolved(sent.pipeline).pipeline.source.options.stride_fraction == 1.0  # type: ignore[attr-defined]
    assert sent.invocation.output_path == "out.npz"
    assert sent.invocation.configuration == {
        "min_tissue_fraction": 0.0,
        "skip_background": True,
        "stride_fraction": 1.0,
    }


@pytest.mark.parametrize(
    ("path", "changes", "status", "code"),
    [
        ("/v1/pipelines/missing/run", {}, 404, "unknown_pipeline"),
        (RUN, {"configuration": {"model_id": "x"}}, 422, "invalid_request"),
        (RUN, {"configuration": {"stride_fraction": -1}}, 422, "invalid_request"),
        (RUN, {"configuration": {"stride_fraction": 0.01}}, 422, "invalid_request"),
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
        (RUN, {"deadline_at": "2100-01-01T00:00:00"}, 422, "invalid_request"),
    ],
)
async def test_a_run_the_pipeline_cannot_take_is_refused_before_dispatch(
    path: str, changes: dict[str, object], status: int, code: str
) -> None:
    runner = FakeRunner()

    answered, body = await post(runner, path, **changes)

    assert (answered, body["detail"]["code"]) == (status, code)  # type: ignore[index]
    assert runner.requests == []


async def test_an_invalid_body_names_the_field() -> None:
    _, body = await post(FakeRunner(), deadline_at="2100-01-01T00:00:00")

    assert "deadline_at" in body["detail"]["message"]  # type: ignore[index]


async def test_an_unknown_setting_is_named() -> None:
    _, body = await post(FakeRunner(), configuration={"model_id": "x"})

    assert "model_id" in body["detail"]["message"]  # type: ignore[index]


@pytest.mark.parametrize(
    ("error", "status", "code", "retryable"),
    [
        (
            RemoteFailureError(
                code="invalid_model_output",
                safe_message="model returned an invalid output",
                retryable=False,
            ),
            422,
            "invalid_model_output",
            False,
        ),
        (
            RemoteFailureError(
                code="model_failed", safe_message="down", retryable=True
            ),
            503,
            "model_failed",
            True,
        ),
        (
            ConfirmedRejectionError(timedelta(seconds=7)),
            429,
            "pipeline_runner_backpressure",
            True,
        ),
        (UnknownOutcomeError("lost"), 503, "pipeline_runner_unavailable", True),
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

    assert (status, body["detail"]["code"]) == (503, "pipeline_runner_unavailable")  # type: ignore[index]


async def test_models_list_models_with_declarations_and_pipelines() -> None:
    served = CATALOG.models
    app = create_app(NoScheduler(), catalog=lambda: CATALOG)  # type: ignore[arg-type]
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://model-service"
    ) as client:
        listed = (await client.get("/v1/models")).json()
        single = (await client.get("/v1/models/prostate-classifier-1")).json()
        missing = await client.get("/v1/models/missing")

    by_id = {model["id"]: model for model in listed}
    assert sorted(by_id) == sorted(served)
    prostate = by_id["prostate-classifier-1"]
    assert single == prostate
    assert set(prostate) == {"id", "declaration", "digest", "pipelines"}
    assert (missing.status_code, missing.json()["detail"]["code"]) == (
        404,
        "unknown_model",
    )
    assert "prostate-heatmap" in prostate["pipelines"]
    assert prostate["declaration"]["input"] == {"tile_size": 512, "mpp": 0.468}
    # The environment's patch is part of the declaration.
    assert prostate["declaration"]["loader"]["uri"].startswith("mlflow-artifacts:/169/")
    assert prostate["digest"] == served["prostate-classifier-1"].digest
    assert "virchow2-embeddings" in by_id["virchow2"]["pipelines"]
