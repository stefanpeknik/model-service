"""Each process imports only its own packages: the API never loads Ray, replicas never load the API."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"


def loaded_modules(module: str) -> set[str]:
    """Import `module` in a fresh interpreter and list every module it loaded."""
    script = f"import json, sys, {module}; print(json.dumps(sorted(sys.modules)))"
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    )
    return set(json.loads(result.stdout.splitlines()[-1]))


def under(modules: set[str], *packages: str) -> list[str]:
    """The modules that are one of `packages` or inside them."""
    return sorted(
        m for m in modules for p in packages if m == p or m.startswith(p + ".")
    )


class ProcessBoundaryTests(unittest.TestCase):
    def test_the_api_process_loads_no_ray_and_no_replica_code(self) -> None:
        modules = loaded_modules("model_service.api.app_factory")

        self.assertEqual(
            under(
                modules,
                "ray",
                "model_service.builders.replica",
                "model_service.builders.model_clients",
                "model_service.inference",
            ),
            [],
        )

    def test_builder_replicas_load_no_api_code(self) -> None:
        modules = loaded_modules("model_service.builders.replica.serve")

        self.assertEqual(under(modules, "model_service.api"), [])


if __name__ == "__main__":
    unittest.main()
