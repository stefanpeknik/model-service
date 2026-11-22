"""The Serve application a served-model file deploys as, which Helm renders."""

from __future__ import annotations

from pathlib import Path

from model_service.catalog import load_catalog
from model_service.served_model.deployment import (
    DEFAULT_CACHE_ROOT,
    host_config,
    serve_application,
)
from tests.support.model_documents import model_file


REPO = Path(__file__).resolve().parents[2]
CATALOG = REPO / "catalog"
OLD_MODELS = REPO / "deploy" / "helm" / "rayservice" / "models"


def test_episeg_deploys_with_the_serve_options_of_its_old_model_file() -> None:
    model_id = "episeg-1"
    entry = load_catalog(CATALOG).entries("served-models")[model_id]
    old = model_file(OLD_MODELS / f"{model_id}.yaml")["deployment"]
    del old["user_config"], old["ray_actor_options"]["runtime_env"]

    app = serve_application(entry, cache_root="/mnt/cache/trt_cache")
    (deployment,) = app.pop("deployments")
    user_config = deployment.pop("user_config")

    assert app == {
        "name": model_id,
        "route_prefix": f"/{model_id}",
        "import_path": "model_service.served_model.serve:app",
    }
    assert deployment == {"name": "ServedModel", **old}
    assert user_config == {
        "id": model_id,
        "declaration": entry.document,
        "cache_root": "/mnt/cache/trt_cache",
    }


def test_the_host_reads_back_the_model_the_catalog_resolved() -> None:
    entry = load_catalog(CATALOG).entries("served-models")["prostate-classifier-1"]
    (deployment,) = serve_application(entry, cache_root="/cache")["deployments"]

    model, cache_root = host_config(deployment["user_config"])
    _, default_root = host_config(
        {k: v for k, v in deployment["user_config"].items() if k != "cache_root"}
    )

    assert (model.id, model.digest) == ("prostate-classifier-1", entry.value.digest)
    assert (cache_root, default_root) == (Path("/cache"), Path(DEFAULT_CACHE_ROOT))


def test_serving_limits_follow_the_batch_size_unless_given(tmp_path: Path) -> None:
    root = tmp_path / "catalog"
    (root / "served-models").mkdir(parents=True)
    (root / "served-models" / "m1.yaml").write_text(
        "weights: {use: local, path: /m.onnx}\n"
        "runtime: {use: onnx}\n"
        "input: {tile_size: 64, mpp: 0.5}\n"
        "output: {kind: scores}\n"
        "batching: {max_size: 8, wait_s: 0}\n"
        "resources: {gpus: 0.5, memory_gib: 1.5}\n"
        "serving: {max_replicas: 1, max_queued_requests: 10}\n"
    )
    entry = load_catalog(root).entries("served-models")["m1"]

    (deployment,) = serve_application(entry, cache_root="/c")["deployments"]

    assert {k: v for k, v in deployment.items() if k != "user_config"} == {
        "name": "ServedModel",
        "max_ongoing_requests": 16,
        "max_queued_requests": 10,
        "autoscaling_config": {
            "min_replicas": 0,
            "max_replicas": 1,
            "target_ongoing_requests": 4,
        },
        "ray_actor_options": {"num_cpus": 1, "num_gpus": 0.5, "memory": 1610612736},
    }
