"""Check facade settings parsing from the environment."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from model_service.facade.settings import SchedulerSettings


class SchedulerSettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = {"RAY_SERVE_ORIGIN": "http://builder-serve"}

    def test_environment_lists_paths_and_seconds_are_parsed(self) -> None:
        settings = SchedulerSettings.from_environment(
            {
                **self.environment,
                "MODEL_CATALOG_PATH": "/etc/catalog.json",
                "BUILDER_IDS": '["heatmap"]',
                "INFERENCE_SERVE_ORIGIN": " http://model-serve ",
                "INFERENCE_MODEL_ORIGINS": '{"classifier":"http://model-serve"}',
                "INFERENCE_TIMEOUT_SECONDS": "12.5",
                "UNRELATED_SETTING": "ignored",
            }
        )

        self.assertEqual(settings.model_catalog_path, Path("/etc/catalog.json"))
        self.assertEqual(settings.builder_ids, {"heatmap"})
        self.assertEqual(settings.inference_serve_origin, "http://model-serve")
        self.assertEqual(
            str(settings.inference_model_origins["classifier"]), "http://model-serve/"
        )
        self.assertEqual(settings.inference_timeout.total_seconds(), 12.5)

    def test_unset_optional_settings_deploy_all_builders(self) -> None:
        settings = SchedulerSettings.from_environment(
            {**self.environment, "INFERENCE_SERVE_ORIGIN": " \t "}
        )

        self.assertIsNone(settings.builder_ids)
        self.assertIsNone(settings.inference_serve_origin)

    def test_explicit_mapping_never_falls_back_to_process_environment(self) -> None:
        with patch.dict(os.environ, self.environment, clear=True):
            self.assertEqual(
                SchedulerSettings.from_environment().ray_serve_origin,
                "http://builder-serve",
            )
            with self.assertRaises(ValidationError):
                SchedulerSettings.from_environment({})

    def test_invalid_values_are_rejected(self) -> None:
        for overrides in (
            {"RAY_SERVE_ORIGIN": " "},
            {"BUILDER_IDS": "[]"},
            {"BUILDER_IDS": "heatmap"},
            {"INFERENCE_TIMEOUT_SECONDS": "nan"},
            {"INFERENCE_MAX_REQUEST_BYTES": "0"},
            {"INFERENCE_MODEL_ORIGINS": '{"nested/model":"http://model-serve"}'},
            {"INFERENCE_MODEL_ORIGINS": '{"model":"file:///tmp/model"}'},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                SchedulerSettings.from_environment({**self.environment, **overrides})


if __name__ == "__main__":
    unittest.main()
