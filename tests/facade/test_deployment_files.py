"""The Helm values, model files and registered builders agree, before anything deploys.

Each values file is checked as the chart reads it: its `models` become the
facade's catalog and its `builders` become Ray applications. Mistakes here
otherwise surface only when the facade starts.
"""

from __future__ import annotations

import unittest

from model_service.builders.registry import BUILDERS
from tests.support.deployment import CHART, VALUES, load


class DeploymentFileTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
