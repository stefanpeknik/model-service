import unittest
from pathlib import Path

import yaml

from workflow_execution.catalog import WorkflowCatalog


class WorkflowDeploymentTests(unittest.TestCase):
    def test_repo_catalog_matches_enabled_helm_applications(self) -> None:
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
