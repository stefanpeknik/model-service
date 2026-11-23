"""What the Helm chart renders from `catalog/`, checked with `helm template`."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from model_service.api.catalog_mount import MountedCatalog
from model_service.catalog import load_catalog
from model_service.served_model.deployment import serve_application


REPO = Path(__file__).resolve().parents[2]
CHART = REPO / "deploy" / "helm" / "rayservice"
CATALOG = REPO / "catalog"
TEST_VALUES = CHART / "values.rationai-test.yaml"
TEST_ANALYSES = ("prostate-heatmap", "virchow2-embeddings", "prov-gigapath-embeddings")
BASE_VALUES = CHART / "values.yaml"
TEST_MLFLOW = "https://mlflow.rationai.cloud.e-infra.cz"
BASE_MLFLOW = "http://mlflow.rationai-mlflow:5000"

pytestmark = pytest.mark.skipif(shutil.which("helm") is None, reason="needs helm")


def render(*values: Path, **settings: object) -> dict[tuple[str, str], Any]:
    """Render the chart and index its objects by kind and name.

    Args:
        values: Values files, in order.
        settings: `--set-json` overrides by key.

    Returns:
        Every rendered object, keyed by `(kind, metadata.name)`.
    """
    command = ["helm", "template", "model-service", str(CHART)]
    for path in values:
        command += ["-f", str(path)]
    for key, value in settings.items():
        command += ["--set-json", f"{key}={json.dumps(value)}"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    objects = [o for o in yaml.safe_load_all(result.stdout) if o]
    return {(o["kind"], o["metadata"]["name"]): o for o in objects}


def applications(objects: dict[tuple[str, str], Any]) -> dict[str, Any]:
    """The Serve applications of the rendered RayService, by name."""
    service = objects[("RayService", "model-service")]
    config = yaml.safe_load(service["spec"]["serveConfigV2"])
    return {app["name"]: app for app in config["applications"]}


def cpu_worker_resource() -> dict[str, Any]:
    """The base values' `cpuWorkerResource`: the custom resource keeping CPU-only apps off GPU workers."""
    return yaml.safe_load(BASE_VALUES.read_text(encoding="utf-8"))["cpuWorkerResource"]


def catalog_data(objects: dict[tuple[str, str], Any]) -> dict[str, Any]:
    """The API's catalog ConfigMap, each key decoded from JSON."""
    data = objects[("ConfigMap", "model-service-model-catalog")]["data"]
    return {key: json.loads(text) for key, text in data.items()}


@pytest.mark.parametrize(
    ("values", "environment", "model_id", "mlflow"),
    [
        ((TEST_VALUES,), "rationai-test", "prostate-classifier-1", TEST_MLFLOW),
        ((TEST_VALUES,), "rationai-test", "virchow2", TEST_MLFLOW),
        ((), None, "prostate-classifier-1", BASE_MLFLOW),
        ((), None, "episeg-1", BASE_MLFLOW),
        ((), None, "virchow2", BASE_MLFLOW),
    ],
)
def test_a_served_model_deploys_as_serve_application_builds_it(
    values: tuple[Path, ...], environment: str | None, model_id: str, mlflow: str
) -> None:
    entry = load_catalog(CATALOG, environment).served_models[model_id]
    expected = serve_application(
        entry, cache_root="/mnt/cache/trt_cache", setup_timeout_seconds=1800
    )
    expected["deployments"][0]["ray_actor_options"]["runtime_env"] = {
        "env_vars": {"MLFLOW_TRACKING_URI": mlflow, "HF_HOME": "/mnt/huggingface_cache"}
    }

    assert applications(render(*values))[model_id] == expected


def test_an_environment_deploys_its_served_models_and_one_runner() -> None:
    apps = applications(render(TEST_VALUES))

    assert set(apps) == {
        "prostate-classifier-1",
        "virchow2",
        "prov-gigapath",
        "analysis-runner",
    }
    assert {
        apps[i]["import_path"]
        for i in ("prostate-classifier-1", "virchow2", "prov-gigapath")
    } == {"model_service.served_model.serve:app"}


def test_the_base_values_serve_every_model_on_the_generic_host() -> None:
    apps = applications(render())

    assert set(apps) == {
        *load_catalog(CATALOG).served_models,
        "analysis-runner",
    }
    assert {
        app["import_path"] for app in apps.values() if app["name"] != "analysis-runner"
    } == {"model_service.served_model.serve:app"}


def test_the_api_catalog_carries_the_deployed_served_models() -> None:
    data = catalog_data(render(TEST_VALUES))
    catalog = load_catalog(CATALOG, "rationai-test")
    entries = catalog.served_models
    analyses = catalog.analyses

    assert data["catalog.json"] == {
        "served-models": {
            model_id: entries[model_id].declaration
            for model_id in ("prostate-classifier-1", "virchow2", "prov-gigapath")
        },
        "analyses": {
            analysis_id: analyses[analysis_id].declaration
            for analysis_id in TEST_ANALYSES
        },
    }
    # The catalog is all the API reads.
    assert set(data) == {"catalog.json"}
    assert "empaia" in data["catalog.json"]["analyses"]["prostate-heatmap"]["expose"]


def test_the_api_reads_the_rendered_catalog(tmp_path: Path) -> None:
    data = render(TEST_VALUES)[("ConfigMap", "model-service-model-catalog")]["data"]
    for name, text in data.items():
        (tmp_path / name).write_text(text, encoding="utf-8")

    catalog = MountedCatalog(tmp_path)

    served = catalog.resolved.served_models
    assert sorted(served) == ["prostate-classifier-1", "prov-gigapath", "virchow2"]
    assert served["prostate-classifier-1"].input.tile_size == 512
    prostate = catalog.resolved.analyses["prostate-heatmap"]
    assert prostate.empaia is not None
    assert sorted(catalog.resolved.analyses) == sorted(TEST_ANALYSES)


def test_an_environment_deploys_a_named_subset_or_none() -> None:
    none = render(TEST_VALUES, servedModels=[], analyses=[])
    named = render(
        TEST_VALUES,
        servedModels=["prostate-classifier-1"],
        analyses=["prostate-heatmap"],
    )

    assert "prostate-classifier-1" not in applications(none)
    assert catalog_data(none)["catalog.json"] == {"served-models": {}, "analyses": {}}
    assert "prostate-classifier-1" in applications(named)


def test_a_named_served_model_without_a_file_fails_the_render() -> None:
    with pytest.raises(subprocess.CalledProcessError) as failed:
        render(TEST_VALUES, servedModels=["no-such-model"])

    assert "no-such-model" in failed.value.stderr


def test_catalog_changes_reach_the_api_without_rolling_it() -> None:
    api = render(TEST_VALUES)[("Deployment", "model-service-api")]
    pod = api["spec"]["template"]
    container = pod["spec"]["containers"][0]
    (mount,) = [m for m in container["volumeMounts"] if m["name"] == "model-catalog"]
    (volume,) = [v for v in pod["spec"]["volumes"] if v["name"] == "model-catalog"]
    env = {e["name"]: e.get("value") for e in container["env"]}
    assert "annotations" not in pod["metadata"]
    # kubelet updates a ConfigMap volume in place only without subPath or items.
    assert "subPath" not in mount
    assert "items" not in volume["configMap"]
    assert env["MODEL_CATALOG_DIR"] == mount["mountPath"]


def test_one_analysis_runner_serves_the_deployed_analyses() -> None:
    values = yaml.safe_load(TEST_VALUES.read_text(encoding="utf-8"))
    sizing = values["analysisRunner"]
    marker = cpu_worker_resource()
    actor = {
        **sizing["deployment"]["ray_actor_options"],
        "resources": {marker["name"]: marker["perReplica"]},
    }

    runner = applications(render(TEST_VALUES))["analysis-runner"]

    assert runner == {
        "name": "analysis-runner",
        "route_prefix": "/analyses",
        "import_path": "model_service.analysis_runner.serve:app",
        "deployments": [
            {
                "name": "AnalysisRunner",
                **sizing["deployment"],
                "ray_actor_options": actor,
                "user_config": {
                    "shared_storage_roots": ["/mnt/shared"],
                    **sizing["userConfig"],
                },
            }
        ],
    }


def test_without_analyses_there_is_no_runner() -> None:
    objects = render(TEST_VALUES, analyses=[])

    assert "analysis-runner" not in applications(objects)
    assert catalog_data(objects)["catalog.json"]["analyses"] == {}


def test_the_base_values_deploy_every_analysis() -> None:
    data = catalog_data(render())

    assert set(data["catalog.json"]["analyses"]) == set(load_catalog(CATALOG).analyses)


def test_an_analysis_calling_an_undeployed_served_model_fails_the_render() -> None:
    with pytest.raises(subprocess.CalledProcessError) as failed:
        render(TEST_VALUES, servedModels=["virchow2"])

    assert "prostate-heatmap" in failed.value.stderr
    assert "prostate-classifier-1" in failed.value.stderr


def test_a_named_analysis_without_a_file_fails_the_render() -> None:
    with pytest.raises(subprocess.CalledProcessError) as failed:
        render(TEST_VALUES, analyses=["no-such-analysis"])

    assert "no-such-analysis" in failed.value.stderr


@pytest.mark.parametrize("values", [(TEST_VALUES,), ()])
def test_cpu_only_apps_can_only_be_placed_on_cpu_workers(
    values: tuple[Path, ...],
) -> None:
    # The autoscaler's own reading of each worker group, rayStartParams quoting included.
    kuberay = pytest.importorskip("ray.autoscaler._private.kuberay.autoscaling_config")
    name = cpu_worker_resource()["name"]
    objects = render(*values)
    service = objects[("RayService", "model-service")]
    groups = {
        group["groupName"]: kuberay._get_ray_resources_from_group_spec(group, False)
        for group in service["spec"]["rayClusterConfig"]["workerGroupSpecs"]
    }
    apps = applications(objects)
    cpu_only = [
        deployment
        for app_name, app in apps.items()
        if app_name == "analysis-runner"
        for deployment in app["deployments"]
    ]
    served = [
        deployment
        for app in apps.values()
        if app["import_path"] == "model_service.served_model.serve:app"
        for deployment in app["deployments"]
    ]
    requested = [
        d["ray_actor_options"].get("resources", {}).get(name, 0) for d in cpu_only
    ]
    co_located = sum(
        amount * d["autoscaling_config"]["max_replicas"]
        for amount, d in zip(requested, cpu_only, strict=True)
    )
    cpu_groups = {g: r for g, r in groups.items() if not r.get("GPU")}
    gpu_groups = {g: r for g, r in groups.items() if r.get("GPU")}

    assert "analysis-runner" in apps
    assert cpu_groups and gpu_groups
    assert all(amount > 0 for amount in requested)
    assert all(
        resources.get(name, 0) >= co_located for resources in cpu_groups.values()
    )
    assert not any(name in resources for resources in gpu_groups.values())
    assert not any(name in d["ray_actor_options"].get("resources", {}) for d in served)
