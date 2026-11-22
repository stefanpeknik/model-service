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
from model_service.contracts.model_declarations import SpatialScoresModel
from model_service.served_model.deployment import serve_application


REPO = Path(__file__).resolve().parents[2]
CHART = REPO / "deploy" / "helm" / "rayservice"
CATALOG = REPO / "catalog"
TEST_VALUES = CHART / "values.rationai-test.yaml"
OLD_MODELS = CHART / "models"
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
    entry = load_catalog(CATALOG, environment).entries("served-models")[model_id]
    expected = serve_application(
        entry, cache_root="/mnt/cache/trt_cache", setup_timeout_seconds=1800
    )
    expected["deployments"][0]["ray_actor_options"]["runtime_env"] = {
        "env_vars": {"MLFLOW_TRACKING_URI": mlflow, "HF_HOME": "/mnt/huggingface_cache"}
    }

    assert applications(render(*values))[model_id] == expected


@pytest.mark.parametrize(
    ("values", "old_file"),
    [
        ((TEST_VALUES,), OLD_MODELS / "rationai-test" / "virchow2.yaml"),
        ((), OLD_MODELS / "virchow2.yaml"),
    ],
)
def test_virchow2_keeps_the_serve_options_of_its_old_model_file(
    values: tuple[Path, ...], old_file: Path
) -> None:
    old = yaml.safe_load(old_file.read_text(encoding="utf-8"))
    app = applications(render(*values))["virchow2"]
    (deployment,) = app["deployments"]
    env = deployment["ray_actor_options"]["runtime_env"]["env_vars"]

    assert app["runtime_env"] == old["application"]["runtime_env"]
    assert env["HF_HOME"] == "/mnt/huggingface_cache"
    del env["MLFLOW_TRACKING_URI"], deployment["name"], deployment["user_config"]
    del old["deployment"]["user_config"]
    assert deployment == old["deployment"]


def test_an_environment_without_old_models_deploys_only_the_generic_host() -> None:
    apps = applications(render(TEST_VALUES))

    assert set(apps) == {
        "prostate-classifier-1",
        "virchow2",
        "builder-heatmap",
        "builder-tile-embeddings",
    }
    assert {apps[i]["import_path"] for i in ("prostate-classifier-1", "virchow2")} == {
        "model_service.served_model.serve:app"
    }


def test_the_base_values_still_deploy_the_other_models_the_old_way() -> None:
    apps = applications(render())

    assert apps["prov-gigapath"]["import_path"] == (
        "model_service.inference.implementations.prov_gigapath:app"
    )


def test_the_api_catalog_carries_the_deployed_served_models() -> None:
    data = catalog_data(render(TEST_VALUES))
    entries = load_catalog(CATALOG, "rationai-test").entries("served-models")

    assert data["catalog.json"] == {
        "served-models": {
            model_id: entries[model_id].document
            for model_id in ("prostate-classifier-1", "virchow2")
        }
    }
    assert data["model-catalog.json"] == []
    assert set(data["provider-blocks.json"]["prostate-classifier-1"]) == {"empaia"}


def test_the_api_reads_the_rendered_catalog(tmp_path: Path) -> None:
    data = render(TEST_VALUES)[("ConfigMap", "model-service-model-catalog")]["data"]
    for name, text in data.items():
        (tmp_path / name).write_text(text, encoding="utf-8")

    catalog = MountedCatalog(tmp_path, provider_blocks=["empaia"])

    assert catalog.get("prostate-classifier-1") == SpatialScoresModel(
        id="prostate-classifier-1",
        tile_size=512,
        output_tile_size=1,
        n_channels=1,
        mpp=0.468,
    )
    assert catalog.get("virchow2").contract == "tile-embedding"
    assert set(catalog.provider_block("empaia")) == {"prostate-classifier-1"}


def test_an_environment_deploys_a_named_subset_or_none() -> None:
    none = render(TEST_VALUES, servedModels=[])
    named = render(TEST_VALUES, servedModels=["prostate-classifier-1"])

    assert "prostate-classifier-1" not in applications(none)
    assert catalog_data(none)["catalog.json"] == {"served-models": {}}
    assert catalog_data(none)["provider-blocks.json"] == {}
    assert "prostate-classifier-1" in applications(named)


def test_a_named_served_model_without_a_file_fails_the_render() -> None:
    with pytest.raises(subprocess.CalledProcessError) as failed:
        render(TEST_VALUES, servedModels=["no-such-model"])

    assert "no-such-model" in failed.value.stderr


def test_a_model_deployed_both_ways_fails_the_render() -> None:
    with pytest.raises(subprocess.CalledProcessError) as failed:
        render(TEST_VALUES, models=["prostate-classifier-1"])

    assert "prostate-classifier-1" in failed.value.stderr


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
