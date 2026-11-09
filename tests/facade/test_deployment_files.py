"""The Helm values, model files and registered builders agree, before anything deploys.

Each values file is checked as the chart reads it: its `models` become the
facade's catalog and its `builders` become Ray applications. Mistakes here
otherwise surface only when the facade starts.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from model_service.builders.registry import BUILDERS
from model_service.facade.catalog import ModelCatalog
from model_service.providers.empaia.apps import app_definitions, parse_listings
from model_service.providers.empaia.composition import CATALOG_BLOCK
from model_service.providers.empaia.publishers import default_publishers


CHART = Path(__file__).parents[2] / "deploy/helm/rayservice"
VALUES = sorted(CHART.glob("values*.yaml"))


def load(path: Path) -> dict[str, object]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


class DeploymentFileTests(unittest.TestCase):
    def test_values_files_exist(self) -> None:
        self.assertIn(CHART / "values.yaml", VALUES)

    def test_every_listed_model_has_a_file(self) -> None:
        for values in VALUES:
            for model in load(values).get("models") or []:
                with self.subTest(values=values.name, model=model):
                    self.assertTrue((CHART / f"models/{model}.yaml").is_file())

    def test_deployed_builders_are_exactly_the_registered_ones(self) -> None:
        for values in VALUES:
            builders = load(values).get("builders")
            if builders is None:
                continue  # Inherits the defaults' builders.
            with self.subTest(values.name):
                self.assertEqual(set(builders), set(BUILDERS))

    def test_each_values_file_yields_a_valid_catalog_and_its_apps(self) -> None:
        publishers = default_publishers()
        for values in VALUES:
            models = load(values).get("models")
            if not models:
                continue
            with self.subTest(values.name):
                documents = [load(CHART / f"models/{m}.yaml") for m in models]
                catalog = ModelCatalog.from_documents(documents, [CATALOG_BLOCK])
                listings = parse_listings(catalog.provider_block(CATALOG_BLOCK))
                apps = app_definitions(catalog, listings, publishers)
                # A model listed for EMPAIA that no builder can serve there would
                # otherwise only be logged at startup.
                offered = {app.model_id for app in apps}
                self.assertEqual(offered, set(listings))


if __name__ == "__main__":
    unittest.main()
