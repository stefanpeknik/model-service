"""What the Helm chart renders from `catalog/`, checked with `helm template`."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest
import yaml

from model_service.catalog import load_catalog
from model_service.control_plane.catalog_mount import MountedCatalog
from tests.support.paths import RATIONAI_TEST


REPO = Path(__file__).resolve().parents[2]
CHART = REPO / "deploy" / "helm"
CATALOG = REPO / "catalog"
TEST_VALUES = CHART / "environments" / "rationai-test" / "values.yaml"
SENSITIVE_VALUES = CHART / "environments" / "sensitive-cluster" / "values.yaml"
# What sensitive-cluster leaves for whoever installs it to fill in.
SENSITIVE_PLACEHOLDERS = {
    "rayImage.tag": "t",
    "rayGpuImage.tag": "t",
    "controlPlane.image.tag": "t",
    "controlPlane.database.existingSecret": "model-service-db",
    "controlPlane.encryption.existingSecret": "model-service-db",
}
TEST_ANALYSES = ("prostate-heatmap", "virchow2-embeddings", "prov-gigapath-embeddings")
BASE_VALUES = CHART / "values.yaml"

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
    """The control plane's catalog ConfigMap, each key decoded from JSON."""
    data = objects[("ConfigMap", "model-service-model-catalog")]["data"]
    return {key: json.loads(text) for key, text in data.items()}


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


def sensitive(**settings: object) -> dict[tuple[str, str], Any]:
    """Render sensitive-cluster with its placeholders filled in."""
    return render(SENSITIVE_VALUES, **SENSITIVE_PLACEHOLDERS, **settings)


def test_sensitive_cluster_names_each_placeholder_it_needs() -> None:
    for key in SENSITIVE_PLACEHOLDERS:
        filled = {k: v for k, v in SENSITIVE_PLACEHOLDERS.items() if k != key}
        with pytest.raises(subprocess.CalledProcessError) as failed:
            render(SENSITIVE_VALUES, **filled)
        assert f"{key} is required" in failed.value.stderr


def test_a_release_needs_an_environment() -> None:
    with pytest.raises(subprocess.CalledProcessError) as failed:
        render(workers=["cpu"])

    assert "environment is required" in failed.value.stderr


def test_sensitive_cluster_serves_every_model_on_the_generic_host() -> None:
    apps = applications(sensitive())

    assert set(apps) == {
        *load_catalog(CATALOG).served_models,
        "analysis-runner",
    }
    assert {
        app["import_path"] for app in apps.values() if app["name"] != "analysis-runner"
    } == {"model_service.served_model.serve:app"}


def test_the_control_plane_catalog_carries_the_deployed_served_models() -> None:
    data = catalog_data(render(TEST_VALUES))
    catalog = load_catalog(CATALOG, RATIONAI_TEST)
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
    # The catalog is all the control plane reads.
    assert set(data) == {"catalog.json"}
    assert "empaia" in data["catalog.json"]["analyses"]["prostate-heatmap"]["expose"]


def test_the_control_plane_reads_the_rendered_catalog(tmp_path: Path) -> None:
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


def test_catalog_changes_reach_the_control_plane_without_rolling_it() -> None:
    api = render(TEST_VALUES)[("Deployment", "model-service-control-plane")]
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


def test_sensitive_cluster_deploys_every_analysis() -> None:
    data = catalog_data(sensitive())

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


@pytest.mark.parametrize("environment", ["rationai-test", "sensitive-cluster"])
def test_cpu_only_apps_can_only_be_placed_on_cpu_workers(environment: str) -> None:
    # The autoscaler's own reading of each worker group, rayStartParams quoting included.
    kuberay = pytest.importorskip("ray.autoscaler._private.kuberay.autoscaling_config")
    name = cpu_worker_resource()["name"]
    objects = render(TEST_VALUES) if environment == "rationai-test" else sensitive()
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
