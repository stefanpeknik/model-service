"""Check API settings parsing from the environment."""

from __future__ import annotations

import os
import unittest
from datetime import timedelta
from pathlib import Path
from unittest.mock import patch

from model_service.scheduler.settings import SchedulerSettings


def load(environment: dict[str, str]) -> SchedulerSettings:
    with patch.dict(os.environ, environment, clear=True):
        return SchedulerSettings()


class SchedulerSettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = {
            "DATABASE_URL": "postgresql://unused",
            "PROVIDER_CONTEXT_KEY": "synthetic-key",
            "RAY_SERVE_ORIGIN": "http://builder-serve",
        }

    def test_environment_lists_paths_and_seconds_are_parsed(self) -> None:
        settings = load(
            {
                **self.environment,
                "MODEL_CATALOG_PATH": "/etc/catalog.json",
                "BUILDER_IDS": '["heatmap"]',
                "SHARED_STORAGE_ROOTS": '["/mnt/data", "/mnt/projects"]',
                "LEASE_SECONDS": "12.5",
                "PER_MODEL_LIMIT": "2",
                "MAX_CONCURRENT_EXECUTIONS": "8",
                "UNRELATED_SETTING": "ignored",
            }
        )

        self.assertEqual(settings.model_catalog_path, Path("/etc/catalog.json"))
        self.assertEqual(settings.builder_ids, {"heatmap"})
        self.assertEqual(
            settings.shared_storage_roots, (Path("/mnt/data"), Path("/mnt/projects"))
        )
        self.assertEqual(settings.lease_duration, timedelta(seconds=12.5))
        self.assertEqual(settings.per_model_limit, 2)
        self.assertEqual(settings.max_concurrent_executions, 8)

    def test_unset_optional_settings_deny_shared_storage_and_deploy_all_builders(
        self,
    ) -> None:
        settings = load(self.environment)

        self.assertEqual(settings.shared_storage_roots, ())
        self.assertIsNone(settings.builder_ids)

    def test_an_empty_builder_list_deploys_no_builders(self) -> None:
        settings = load({**self.environment, "BUILDER_IDS": "[]"})

        self.assertEqual(settings.builder_ids, frozenset())

    def test_invalid_values_are_rejected(self) -> None:
        for overrides in (
            {"DATABASE_URL": " "},
            {"BUILDER_IDS": "heatmap"},
            {"SHARED_STORAGE_ROOTS": "/mnt/data"},
            {"PER_MODEL_LIMIT": "0"},
            {"MAX_CONCURRENT_EXECUTIONS": "0"},
            {"LEASE_SECONDS": "nan"},
            {"INITIAL_BACKOFF_SECONDS": "301"},
            {"DEFAULT_DEADLINE_SECONDS": "86401"},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                load({**self.environment, **overrides})


if __name__ == "__main__":
    unittest.main()
