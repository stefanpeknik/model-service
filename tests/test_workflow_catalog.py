from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from workflow_execution.catalog import (
    WorkflowCatalog,
    WorkflowDefinition,
    validate_empty_configuration,
)
from workflow_execution.contracts import WorkflowInputKind, WorkflowOutputKind
from workflow_execution.prostate_heatmap import validate_configuration


class WorkflowCatalogTests(unittest.TestCase):
    def test_catalog_fails_closed_when_workflow_validator_is_missing(self) -> None:
        definition = WorkflowDefinition("unregistered", {}, {})
        with self.assertRaisesRegex(ValueError, "configuration validator"):
            WorkflowCatalog((definition,))

    def test_configuration_free_workflow_explicitly_rejects_parameters(self) -> None:
        catalog = WorkflowCatalog(
            (WorkflowDefinition("empty", {}, {}),),
            configuration_validators={"empty": validate_empty_configuration},
        )
        self.assertEqual(catalog.resolve_configuration("empty", {}), {})
        with self.assertRaisesRegex(ValueError, "must be empty"):
            catalog.resolve_configuration("empty", {"unexpected": True})

    def test_yaml_declarations_validate_consumed_fields_and_preserve_metadata(
        self,
    ) -> None:
        definition = {
            "identifier": "probe",
            "inputs": {"source": {"kinds": ["provider"], "empaia": {"type": "wsi"}}},
            "outputs": {"result": {"kind": "inline"}},
            "empaia": {"name": "Probe"},
        }
        with TemporaryDirectory() as directory:
            path = Path(directory) / "probe.yaml"
            path.write_text(json.dumps(definition), encoding="utf-8")
            catalog = WorkflowCatalog.from_directory(
                Path(directory),
                configuration_validators={"probe": validate_empty_configuration},
            )
            loaded = catalog.get("probe")
            self.assertEqual(
                loaded.inputs["source"].kinds, (WorkflowInputKind.PROVIDER,)
            )
            self.assertEqual(loaded.outputs["result"].kind, WorkflowOutputKind.INLINE)
            self.assertEqual(loaded.metadata, {"empaia": {"name": "Probe"}})
            for kinds in [["unknown"], ["provider", "provider"], [], [True]]:
                with self.subTest(kinds=kinds):
                    definition["inputs"]["source"]["kinds"] = kinds
                    path.write_text(json.dumps(definition), encoding="utf-8")
                    with self.assertRaises(ValueError):
                        WorkflowCatalog.from_directory(
                            Path(directory),
                            configuration_validators={
                                "probe": validate_empty_configuration
                            },
                        )

    def test_defaults_and_provider_overrides_do_not_change_resolved_validation(
        self,
    ) -> None:
        catalog = WorkflowCatalog.default()
        resolved = catalog.resolve_configuration(
            "prostate-heatmap",
            {"stride_fraction": 0.25},
            overrides={"output_path": "provider.tiff"},
        )
        self.assertEqual(resolved["output_path"], "provider.tiff")
        self.assertEqual(resolved["stride_fraction"], 0.25)
        resolved["output_tile_width"] = 0
        self.assertEqual(
            catalog.resolve_configuration("prostate-heatmap", {})["output_tile_width"],
            256,
        )
        with self.assertRaises(ValueError):
            catalog.resolve_configuration("prostate-heatmap", {"unknown": True})

    def test_prostate_configuration_rejects_coercion_and_nonfinite_numbers(
        self,
    ) -> None:
        values = WorkflowCatalog.default().resolve_configuration("prostate-heatmap", {})
        for invalid in [True, "0.5", float("inf"), float("nan")]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_configuration({**values, "stride_fraction": invalid})
        with self.assertRaises(ValueError):
            validate_configuration({**values, "output_tile_height": True})
        with self.assertRaises(ValueError):
            validate_configuration({})
