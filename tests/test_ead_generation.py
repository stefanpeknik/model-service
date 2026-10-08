"""Check generated EMPAIA metadata and input-reference validation."""

from __future__ import annotations

import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml
from jsonschema import ValidationError

from scripts.generate_ead import ROOT, generate_ead, generate_file, load_yaml


class EadGenerationTests(unittest.TestCase):
    """Exercise generated EMPAIA metadata and input-reference validation."""

    def setUp(self) -> None:
        """Create isolated fixtures and clients for generated EMPAIA metadata and input-reference validation.

        Returns:
            None: No return value.
        """
        self.definition_path = ROOT / "workflow-definitions" / "prostate-heatmap.yaml"
        self.definition = load_yaml(self.definition_path)

    def test_prostate_ead_is_generated_from_workflow_contract(self) -> None:
        """Verify prostate ead is generated from workflow contract.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
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
        """Verify references must target inputs.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        definition = deepcopy(self.definition)
        definition["outputs"]["heatmap"]["empaia"]["reference"] = "missing"

        with self.assertRaisesRegex(ValueError, "unknown input"):
            generate_ead(definition)

    def test_schema_rejects_invalid_metadata(self) -> None:
        for field, value in (
            ("name", "x"),
            ("name_short", "too-long-name"),
            ("namespace", "not-an-empaia-namespace"),
            ("permissions", {"datatransmissionexternal": "true"}),
        ):
            with self.subTest(field=field):
                definition = deepcopy(self.definition)
                definition["empaia"][field] = value
                with self.assertRaises(ValidationError):
                    generate_ead(definition)

    def test_schema_rejects_invalid_io(self) -> None:
        for field, value in (("type", "unknown-type"), ("unexpected", True)):
            with self.subTest(field=field):
                definition = deepcopy(self.definition)
                definition["outputs"]["heatmap"]["empaia"][field] = value
                with self.assertRaises(ValidationError):
                    generate_ead(definition)

    def test_invalid_generation_preserves_existing_file(self) -> None:
        definition = deepcopy(self.definition)
        definition["empaia"]["namespace"] = "invalid"
        with TemporaryDirectory() as temporary_directory:
            definition_path = Path(temporary_directory) / "workflow.yaml"
            output_path = Path(temporary_directory) / "ead.json"
            definition_path.write_text(yaml.safe_dump(definition), encoding="utf-8")
            output_path.write_text("existing content", encoding="utf-8")

            for check in (False, True):
                with self.subTest(check=check):
                    with self.assertRaises(ValidationError):
                        generate_file(definition_path, output_path, check=check)
                    self.assertEqual(
                        output_path.read_text(encoding="utf-8"), "existing content"
                    )


if __name__ == "__main__":
    unittest.main()
