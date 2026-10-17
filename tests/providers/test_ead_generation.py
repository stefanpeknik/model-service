"""Check EAD generation from app files and builder EMPAIA mappings."""

from __future__ import annotations

import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import yaml
from jsonschema import ValidationError

from model_service.builders.registry import BuilderSpec, OutputSpec, get_builder
from scripts.generate_ead import ROOT, generate_ead, generate_file, load_yaml


if TYPE_CHECKING:
    from model_service.builders.contracts import JsonValue


APP_PATH = (
    ROOT / "deploy" / "helm" / "rayservice" / "empaia-apps" / "prostate-heatmap.yaml"
)


def _with_heatmap_output(**empaia: JsonValue) -> BuilderSpec:
    spec = get_builder("heatmap")
    heatmap = spec.outputs["heatmap"]
    return replace(
        spec,
        outputs={"heatmap": replace(heatmap, empaia={**heatmap.empaia, **empaia})},
    )


class EadGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.app = load_yaml(APP_PATH)

    def test_prostate_ead_comes_from_builder_mappings(self) -> None:
        ead = generate_ead(self.app)

        self.assertEqual(ead["namespace"], "org.empaia.rationai.prostate_cls.v3.0")
        self.assertEqual(ead["io"]["slide"], {"type": "wsi"})
        self.assertEqual(ead["io"]["region_of_interest"]["reference"], "io.slide")
        self.assertEqual(ead["io"]["heatmap"]["type"], "continuous_pixelmap")
        self.assertEqual(
            ead["modes"]["standalone"],
            {"inputs": ["slide", "region_of_interest"], "outputs": ["heatmap"]},
        )
        self.assertTrue(
            generate_file(
                APP_PATH,
                ROOT / "deploy" / "empaia" / "prostate-heatmap.json",
                check=True,
            )
        )

    def test_app_must_name_a_model(self) -> None:
        app = deepcopy(self.app)
        del app["model"]
        with self.assertRaisesRegex(ValueError, "model deployment"):
            generate_ead(app)

    def test_outputs_without_empaia_type_are_rejected(self) -> None:
        app = {**deepcopy(self.app), "builder": "tile-embeddings", "model": "virchow2"}
        with self.assertRaisesRegex(ValueError, "no EMPAIA type"):
            generate_ead(app)

    def test_references_must_target_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown input"):
            generate_ead(self.app, _with_heatmap_output(reference="missing"))

    def test_schema_rejects_invalid_metadata(self) -> None:
        for field, value in (
            ("name", "x"),
            ("name_short", "too-long-name"),
            ("namespace", "not-an-empaia-namespace"),
            ("permissions", {"datatransmissionexternal": "true"}),
        ):
            with self.subTest(field=field):
                app = deepcopy(self.app)
                app["empaia"][field] = value
                with self.assertRaises(ValidationError):
                    generate_ead(app)

    def test_schema_rejects_invalid_io(self) -> None:
        for field, value in (("type", "unknown-type"), ("unexpected", True)):
            with self.subTest(field=field):
                spec = _with_heatmap_output(**{field: value})
                self.assertIsInstance(spec.outputs["heatmap"], OutputSpec)
                with self.assertRaises(ValidationError):
                    generate_ead(self.app, spec)

    def test_invalid_generation_preserves_existing_file(self) -> None:
        app = deepcopy(self.app)
        app["empaia"]["namespace"] = "invalid"
        with TemporaryDirectory() as temporary_directory:
            app_path = Path(temporary_directory) / "app.yaml"
            output_path = Path(temporary_directory) / "ead.json"
            app_path.write_text(yaml.safe_dump(app), encoding="utf-8")
            output_path.write_text("existing content", encoding="utf-8")

            for check in (False, True):
                with self.subTest(check=check):
                    with self.assertRaises(ValidationError):
                        generate_file(app_path, output_path, check=check)
                    self.assertEqual(
                        output_path.read_text(encoding="utf-8"), "existing content"
                    )


if __name__ == "__main__":
    unittest.main()
