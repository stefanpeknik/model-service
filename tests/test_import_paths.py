"""Every module path the chart names as a string resolves to a real attribute.

Code moves rename modules, and string paths escape the type checker: the
`import_path` of each Serve application in the Helm templates. This test
imports each module for real and looks the attribute up. The served-model
host pulls in Ray, ONNX Runtime and torch; the dev environment has them, so a
real import is affordable and catches more than a find_spec check would. The
templates are not YAML, so paths are read with a regular expression.
"""

import importlib
import re
import unittest
from pathlib import Path


TEMPLATES = Path(__file__).resolve().parents[1] / "deploy/helm/rayservice/templates"
PATH = re.compile(r"\bimport_path\W+([\w.]+:\w+)")


def deployment_paths() -> list[str]:
    """Return the module paths the Helm templates name."""
    return [
        path
        for template in sorted(TEMPLATES.glob("*"))
        for path in PATH.findall(template.read_text())
    ]


class ImportPathTests(unittest.TestCase):
    def test_deployment_paths_resolve(self) -> None:
        """Every import_path in the chart resolves."""
        paths = deployment_paths()
        self.assertEqual(
            set(paths),
            {
                "model_service.analysis_runner.serve:app",
                "model_service.served_model.serve:app",
            },
        )
        for path in paths:
            with self.subTest(path=path):
                module_name, attribute = path.split(":")
                module = importlib.import_module(module_name)
                self.assertTrue(hasattr(module, attribute), path)


if __name__ == "__main__":
    unittest.main()
