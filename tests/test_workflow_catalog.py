from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from workflow_execution.catalog import (
    MissingWorkflowApplicationsError,
    WorkflowCatalog,
    WorkflowConfiguration,
    WorkflowDefinition,
    validate_empty_configuration,
)
from workflow_execution.contracts import WorkflowInputKind, WorkflowOutputKind
from workflow_execution.prostate_heatmap import (
    default_configuration,
    validate_configuration,
)


class WorkflowCatalogTests(unittest.TestCase):
    def test_catalog_fails_closed_when_workflow_validator_is_missing(self) -> None:
        definition = WorkflowDefinition("unregistered", {}, {})
        with self.assertRaisesRegex(ValueError, "configuration registration"):
            WorkflowCatalog((definition,), configurations={})

    def test_configuration_free_workflow_explicitly_rejects_parameters(self) -> None:
        catalog = WorkflowCatalog(
            (WorkflowDefinition("empty", {}, {}),),
            configurations={
                "empty": WorkflowConfiguration(validator=validate_empty_configuration)
            },
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
                configurations={
                    "probe": WorkflowConfiguration(
                        validator=validate_empty_configuration
                    )
                },
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
                            configurations={
                                "probe": WorkflowConfiguration(
                                    validator=validate_empty_configuration
                                )
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

    def test_catalog_reports_declared_workflow_without_application(self) -> None:
        catalog = WorkflowCatalog.default()

        with self.assertRaises(MissingWorkflowApplicationsError) as raised:
            catalog.validate_deployed(set())

        self.assertEqual(raised.exception.workflow_ids, ("prostate-heatmap",))

    def test_registration_validates_defaults_at_construction(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be empty"):
            WorkflowCatalog(
                (WorkflowDefinition("empty", {}, {}),),
                configurations={
                    "empty": WorkflowConfiguration(
                        validator=validate_empty_configuration,
                        defaults={"unexpected": True},
                    )
                },
            )

    def test_registration_rejects_configuration_for_undeclared_workflow(self) -> None:
        with self.assertRaisesRegex(ValueError, "configuration registration"):
            WorkflowCatalog(
                (WorkflowDefinition("empty", {}, {}),),
                configurations={
                    workflow_id: WorkflowConfiguration(
                        validator=validate_empty_configuration
                    )
                    for workflow_id in ("empty", "undeclared")
                },
            )

    def test_registration_retains_its_own_copy_of_defaults(self) -> None:
        defaults = default_configuration()
        catalog = WorkflowCatalog(
            (WorkflowCatalog.default().get("prostate-heatmap"),),
            configurations={
                "prostate-heatmap": WorkflowConfiguration(
                    validator=validate_configuration, defaults=defaults
                )
            },
        )
        defaults["output_tile_width"] = 0
        self.assertEqual(
            catalog.resolve_configuration("prostate-heatmap", {})["output_tile_width"],
            256,
        )

    def test_registration_supports_required_configuration_without_defaults(
        self,
    ) -> None:
        catalog = WorkflowCatalog(
            (WorkflowCatalog.default().get("prostate-heatmap"),),
            configurations={
                "prostate-heatmap": WorkflowConfiguration(
                    validator=validate_configuration
                )
            },
        )
        with self.assertRaises(ValueError):
            catalog.resolve_configuration("prostate-heatmap", {})
        values = default_configuration()
        self.assertEqual(
            catalog.resolve_configuration("prostate-heatmap", values), values
        )
