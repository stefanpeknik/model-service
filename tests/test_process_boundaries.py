"""Each process imports only its own packages: the API never loads Ray, replicas never load the API.

The processes: the API (`api/`), the generic served-model host
(`served_model/`) and the analysis runner (`analysis_runner/`); `catalog/`,
`components/`, `formats/` and `contracts/` are shared and stay light.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
CATALOG = SRC.parent / "catalog"


def loaded_modules(module: str, then: str = "") -> set[str]:
    """Import `module` in a fresh interpreter, run `then`, and list every module loaded."""
    script = "; ".join(
        filter(
            None,
            [
                f"import json, sys, {module}",
                then,
                "print(json.dumps(sorted(sys.modules)))",
            ],
        )
    )
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
                "onnxruntime",
                "torch",
                "timm",
                "mlflow",
                "ratiopath",
                "pyvips",
                "openslide",
                "model_service.served_model.serve",
                "model_service.analysis_runner",
                "model_service.slides",
            ),
            [],
        )

    def test_every_component_and_the_catalog_load_without_heavy_libraries(
        self,
    ) -> None:
        """The API validates catalogs, so component declarations must stay light."""
        modules = loaded_modules(
            "model_service.catalog, model_service.components, model_service.testing",
            then=(
                "model_service.components.default_registry(); "
                f"model_service.catalog.load_catalog(__import__('pathlib').Path({str(CATALOG)!r}))"
            ),
        )

        self.assertEqual(
            under(
                modules,
                "ray",
                "onnxruntime",
                "torch",
                "timm",
                "mlflow",
                "openslide",
                "pyvips",
                "ratiopath",
                "h5py",
            ),
            [],
        )

    def test_the_replicas_load_no_api_code(self) -> None:
        for serve in ("analysis_runner", "served_model"):
            with self.subTest(serve):
                modules = loaded_modules(f"model_service.{serve}.serve")
                self.assertEqual(under(modules, "model_service.api"), [])

    def test_the_served_model_replica_loads_no_runner_or_slide_code(self) -> None:
        modules = loaded_modules("model_service.served_model.serve")

        self.assertEqual(
            under(
                modules,
                "model_service.analysis_runner",
                "model_service.slides",
                "openslide",
                "pyvips",
            ),
            [],
        )

    def test_the_served_model_host_core_loads_without_ray(self) -> None:
        """Only the Serve wrapper needs Ray; the chart's tests and tools use the rest."""
        modules = loaded_modules(
            "model_service.served_model.deployment, model_service.served_model.host"
        )

        self.assertEqual(
            under(
                modules, "ray", "onnxruntime", "torch", "mlflow", "model_service.api"
            ),
            [],
        )

    def test_the_analysis_runner_core_loads_without_ray(self) -> None:
        """Only its Serve wrapper may need Ray; slide libraries load per run."""
        modules = loaded_modules("model_service.analysis_runner")

        self.assertEqual(
            under(
                modules,
                "ray",
                "onnxruntime",
                "torch",
                "openslide",
                "pyvips",
                "ratiopath",
                "model_service.api",
            ),
            [],
        )


if __name__ == "__main__":
    unittest.main()
