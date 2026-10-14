"""Check facade settings parsing from the environment."""

from __future__ import annotations

import os
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from model_service.settings import SchedulerSettings


class SchedulerSettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = {
            "DATABASE_URL": "postgresql://unused",
            "PROVIDER_CONTEXT_KEY": "synthetic-key",
            "RAY_SERVE_ORIGIN": "http://builder-serve",
        }

    def test_environment_lists_paths_and_seconds_are_parsed(self) -> None:
        settings = SchedulerSettings.from_environment(
            {
                **self.environment,
                "MODEL_CATALOG_PATH": "/etc/catalog.json",
                "BUILDER_IDS": '["heatmap", "tile-embeddings"]',
                "SHARED_STORAGE_ROOTS": '["/mnt/data", "/mnt/projects"]',
                "INFERENCE_SERVE_ORIGIN": " http://model-serve ",
                "INFERENCE_MODEL_ORIGINS": '{"classifier":"http://model-serve"}',
                "LEASE_SECONDS": "12.5",
                "PER_MODEL_LIMIT": "2",
                "MAX_CONCURRENT_EXECUTIONS": "8",
                "UNRELATED_SETTING": "ignored",
            }
        )

        self.assertEqual(settings.model_catalog_path, Path("/etc/catalog.json"))
        self.assertEqual(settings.builder_ids, {"heatmap", "tile-embeddings"})
        self.assertEqual(
            settings.shared_storage_roots, (Path("/mnt/data"), Path("/mnt/projects"))
        )
        self.assertEqual(settings.inference_serve_origin, "http://model-serve")
        self.assertEqual(
            str(settings.inference_model_origins["classifier"]), "http://model-serve/"
        )
        self.assertEqual(settings.lease_duration, timedelta(seconds=12.5))
        self.assertEqual(settings.per_model_limit, 2)
        self.assertEqual(settings.max_concurrent_executions, 8)

    def test_unset_optional_settings_deny_shared_storage_and_deploy_all_builders(
        self,
    ) -> None:
        settings = SchedulerSettings.from_environment(
            {**self.environment, "INFERENCE_SERVE_ORIGIN": " \t "}
        )

        self.assertEqual(settings.shared_storage_roots, ())
        self.assertIsNone(settings.builder_ids)
        self.assertIsNone(settings.inference_serve_origin)
        self.assertIsNone(settings.empaia_gateway_factory)

    def test_explicit_mapping_never_falls_back_to_process_environment(self) -> None:
        with patch.dict(os.environ, self.environment, clear=True):
            self.assertEqual(
                SchedulerSettings.from_environment().database_url,
                "postgresql://unused",
            )
            with self.assertRaises(ValidationError):
                SchedulerSettings.from_environment({})

    def test_invalid_values_are_rejected(self) -> None:
        for overrides in (
            {"DATABASE_URL": " "},
            {"BUILDER_IDS": "[]"},
            {"BUILDER_IDS": "heatmap"},
            {"SHARED_STORAGE_ROOTS": "/mnt/data"},
            {"PER_MODEL_LIMIT": "0"},
            {"MAX_CONCURRENT_EXECUTIONS": "0"},
            {"LEASE_SECONDS": "nan"},
            {"INFERENCE_MODEL_ORIGINS": '{"nested/model":"http://model-serve"}'},
            {"INFERENCE_MODEL_ORIGINS": '{"model":"file:///tmp/model"}'},
            {"INITIAL_BACKOFF_SECONDS": "301"},
            {"DEFAULT_DEADLINE_SECONDS": "86401"},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                SchedulerSettings.from_environment({**self.environment, **overrides})


if __name__ == "__main__":
    unittest.main()
