"""Check API settings parsing from the environment."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from model_service.scheduler.settings import SchedulerSettings


def load(environment: dict[str, str]) -> SchedulerSettings:
    with patch.dict(os.environ, environment, clear=True):
        return SchedulerSettings()


class SchedulerSettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = {"RAY_SERVE_ORIGIN": "http://builder-serve"}

    def test_environment_paths_and_lists_are_parsed(self) -> None:
        settings = load(
            {
                **self.environment,
                "MODEL_CATALOG_PATH": "/etc/catalog.json",
                "BUILDER_IDS": '["heatmap"]',
                "UNRELATED_SETTING": "ignored",
            }
        )

        self.assertEqual(settings.ray_serve_origin, "http://builder-serve")
        self.assertEqual(settings.model_catalog_path, Path("/etc/catalog.json"))
        self.assertEqual(settings.builder_ids, {"heatmap"})

    def test_unset_builder_selection_deploys_all_builders(self) -> None:
        settings = load(self.environment)

        self.assertIsNone(settings.builder_ids)

    def test_an_empty_builder_list_deploys_no_builders(self) -> None:
        settings = load({**self.environment, "BUILDER_IDS": "[]"})

        self.assertEqual(settings.builder_ids, frozenset())

    def test_invalid_values_are_rejected(self) -> None:
        for environment in (
            {},
            {"RAY_SERVE_ORIGIN": " "},
            {**self.environment, "BUILDER_IDS": "heatmap"},
        ):
            with self.subTest(environment=environment), self.assertRaises(ValueError):
                load(environment)


if __name__ == "__main__":
    unittest.main()
