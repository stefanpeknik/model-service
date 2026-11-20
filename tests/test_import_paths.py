"""Every module path the deployment names as a string resolves to a real attribute.

Code moves rename modules, and string paths escape the type checker: a builder's
spec.implementation, the `implementation` and `_target_` of each model file in
deploy/helm/rayservice/models/ (not rationai-test/, the test instance's copies) and the builder
`import_path` in the Helm template. This test imports each module for real and
looks the attribute up. The model implementations pull in Ray, ONNX Runtime and
torch; the dev environment has them, so a real import is affordable and catches
more than a find_spec check would. The model files are Helm templates, so paths
are read with a regular expression rather than a YAML parser.
"""

import importlib
import re
import unittest
from pathlib import Path

from model_service.builders.registry import BUILDERS
from tests.support.model_documents import implementation, model_file


CHART = Path(__file__).resolve().parents[1] / "deploy" / "helm" / "rayservice"
PATH_LINE = re.compile(r"^\s*(?:import_path|_target_):\s*([\w.]+:\w+)\s*$", re.M)


def deployment_paths() -> list[str]:
    """Return the module paths named by the model files and the Helm template."""
    files = sorted((CHART / "models").glob("*.yaml"))
    files.append(CHART / "templates" / "rayservice.yaml")
    return [path for file in files for path in PATH_LINE.findall(file.read_text())]


class ImportPathTests(unittest.TestCase):
    """Each named module imports and has the named attribute."""

    def assert_resolves(self, path: str) -> None:
        """Import `module:attribute` and fail if either is missing."""
        module_name, attribute = path.split(":")
        module = importlib.import_module(module_name)
        self.assertTrue(hasattr(module, attribute), path)

    def test_builder_implementations_resolve(self) -> None:
        """Every registered builder's implementation string resolves."""
        for builder_id, spec in BUILDERS.items():
            with self.subTest(builder=builder_id):
                self.assert_resolves(spec.implementation)

    def test_model_implementations_resolve(self) -> None:
        """Every model file's implementation has the class and the `app` Ray loads."""
        for path in sorted((CHART / "models").glob("*.yaml")):
            with self.subTest(path.name):
                module, _ = implementation(model_file(path))
                self.assertTrue(hasattr(module, "app"), path.name)

    def test_deployment_paths_resolve(self) -> None:
        """Every import_path and _target_ in the chart resolves."""
        paths = deployment_paths()
        self.assertIn("model_service.builders.replica.serve:app", paths)
        for path in paths:
            with self.subTest(path=path):
                self.assert_resolves(path)
