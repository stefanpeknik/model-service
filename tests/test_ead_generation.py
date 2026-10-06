from __future__ import annotations

import unittest
from copy import deepcopy

from scripts.generate_ead import ROOT, generate_ead, generate_file, load_yaml


class EadGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.definition_path = ROOT / "workflow-definitions" / "prostate-heatmap.yaml"
        self.definition = load_yaml(self.definition_path)

    def test_prostate_ead_is_generated_from_workflow_contract(self) -> None:
        ead = generate_ead(self.definition)

        self.assertEqual(ead["namespace"], "org.empaia.rationai.prostate_cls.v3.0")
        self.assertEqual(ead["io"]["region_of_interest"]["reference"], "io.slide")
        self.assertEqual(ead["modes"]["standalone"]["outputs"], ["heatmap"])
        self.assertNotIn("configuration", ead)
        self.assertNotIn("parameters", self.definition)
        self.assertNotIn("model", self.definition)
        self.assertTrue(
            generate_file(
                self.definition_path,
                ROOT / "eads" / "prostate-heatmap.json",
                check=True,
            )
        )

    def test_references_must_target_inputs(self) -> None:
        definition = deepcopy(self.definition)
        definition["outputs"]["heatmap"]["empaia"]["reference"] = "missing"

        with self.assertRaisesRegex(ValueError, "unknown input"):
            generate_ead(definition)


if __name__ == "__main__":
    unittest.main()
