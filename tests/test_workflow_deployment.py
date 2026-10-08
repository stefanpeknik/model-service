"""Check catalog and Helm deployment configuration agreement."""

import json
import shutil
import subprocess
import unittest
from pathlib import Path

import yaml

from workflow_execution.catalog import WorkflowCatalog


class WorkflowDeploymentTests(unittest.TestCase):
    """Exercise catalog and Helm deployment configuration agreement."""

    @unittest.skipUnless(
        shutil.which("helm"), "Helm is needed to verify rendered deployments"
    )
    def test_full_wsi_preset_joins_native_workflow_and_real_classifier(self) -> None:
        """Verify full WSI preset joins native workflow and real classifier.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        root = Path(__file__).resolve().parents[1]
        rendered = subprocess.run(
            [
                "helm",
                "template",
                "model-service-validation",
                "helm/rayservice",
                "-n",
                "peknik-ns",
                "-f",
                "helm/rayservice/values.full-wsi.yaml",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        manifests = list(yaml.safe_load_all(rendered.stdout))
        ray = next(item for item in manifests if item["kind"] == "RayService")
        applications = yaml.safe_load(ray["spec"]["serveConfigV2"])["applications"]
        self.assertEqual(
            {item["name"] for item in applications},
            {"prostate-heatmap", "prostate-classifier-1"},
        )
        workflow, classifier = applications
        self.assertEqual(
            workflow["import_path"], "workflows.heatmap.prostate_serve:app"
        )
        self.assertEqual(classifier["import_path"], "models.binary_classifier:app")
        self.assertNotIn("working_dir", str(applications))
        model_config = classifier["deployments"][0]["user_config"]
        self.assertEqual(model_config["output_tile_size"], 1)
        self.assertEqual(model_config["n_channels"], 1)
        self.assertEqual(model_config["mpp"], 0.468)
        self.assertEqual(
            model_config["model"]["artifact_uri"],
            "mlflow-artifacts:/169/17dda4dc3f7449478208e87016ce7d86/artifacts/checkpoints/epoch=0-step=6972/prostate_model_norm.onnx",
        )
        worker = ray["spec"]["rayClusterConfig"]["workerGroupSpecs"][0]
        self.assertEqual(worker["minReplicas"], 1)
        self.assertEqual(worker["maxReplicas"], 1)
        worker_resources = worker["template"]["spec"]["containers"][0]["resources"]
        self.assertEqual(worker_resources["requests"]["cpu"], 4)
        self.assertEqual(worker_resources["limits"]["cpu"], 8)
        self.assertEqual(worker_resources["requests"]["memory"], "23Gi")
        self.assertEqual(worker_resources["limits"]["memory"], "24Gi")
        self.assertEqual(
            worker["template"]["spec"]["containers"][0]["resources"]["limits"][
                "nvidia.com/mig-2g.20gb"
            ],
            1,
        )
        self.assertLessEqual(
            sum(
                app["deployments"][0]["ray_actor_options"]["memory"]
                for app in applications
            ),
            23 * 1024**3,
        )
        claims = {
            item["metadata"]["name"]: item
            for item in manifests
            if item["kind"] == "PersistentVolumeClaim"
        }
        self.assertEqual(len(claims), 2)
        for claim in claims.values():
            self.assertEqual(claim["spec"]["resources"]["requests"]["storage"], "32Gi")
            self.assertEqual(
                claim["metadata"]["annotations"]["helm.sh/resource-policy"], "keep"
            )
        scheduler = next(item for item in manifests if item["kind"] == "Deployment")
        scheduler_container = scheduler["spec"]["template"]["spec"]["containers"][0]
        environment = {
            item["name"]: item.get("value") for item in scheduler_container["env"]
        }
        self.assertEqual(environment["DEFAULT_DEADLINE_SECONDS"], "21600")
        self.assertNotIn("EMPAIA_GATEWAY_FACTORY", environment)
        self.assertEqual(
            environment["INFERENCE_SERVE_ORIGIN"],
            "http://peknik-model-service-model-service-serve-svc.peknik-ns.svc.cluster.local:8000",
        )
        self.assertEqual(
            json.loads(environment["INFERENCE_MODEL_ORIGINS"]),
            {
                "prostate-classifier-1": "http://model-service-validation-serve-svc.peknik-ns.svc.cluster.local:8000"
            },
        )
        worker_outputs = next(
            item
            for item in worker["template"]["spec"]["containers"][0]["volumeMounts"]
            if item["name"] == "outputs"
        )
        scheduler_outputs = next(
            item
            for item in scheduler_container["volumeMounts"]
            if item["name"] == "full-wsi-outputs"
        )
        self.assertEqual(worker_outputs["mountPath"], scheduler_outputs["mountPath"])
        self.assertEqual(worker_outputs["mountPath"], "/mnt/validation/outputs")
        for container, volume_name in (
            (scheduler_container, "full-wsi-outputs"),
            (worker["template"]["spec"]["containers"][0], "outputs"),
        ):
            self.assertEqual(
                {
                    mount["mountPath"]
                    for mount in container["volumeMounts"]
                    if mount["name"] == volume_name
                },
                {"/mnt/validation/outputs", "/mnt/validation/.workflow-attempts"},
            )

    def test_repo_catalog_matches_enabled_helm_applications(self) -> None:
        """Verify repo catalog matches enabled helm applications.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        root = Path(__file__).resolve().parents[1]
        chart = root / "helm" / "rayservice"
        values = yaml.safe_load((chart / "values.yaml").read_text(encoding="utf-8"))
        catalog = WorkflowCatalog.default()
        self.assertLessEqual(catalog.identifiers, set(values["applications"]))
        for workflow_id in sorted(catalog.identifiers):
            with self.subTest(workflow_id=workflow_id):
                application = yaml.safe_load(
                    (chart / "applications" / f"{workflow_id}.yaml").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(application[0]["name"], workflow_id)
