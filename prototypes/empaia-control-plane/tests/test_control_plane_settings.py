from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from control_plane.settings import Settings


class SettingsTestCase(unittest.TestCase):
    def test_settings_load_token_file_and_app_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            token_path = Path(temp_dir) / "jes-token"
            token_path.write_text("secret-token\n", encoding="utf-8")
            app_service_token_path = Path(temp_dir) / "app-service-token"
            app_service_token_path.write_text("app-secret-token\n", encoding="utf-8")
            database_path = Path(temp_dir) / "jobs.db"
            mapping = {
                "app_id": "prostate-app",
                "app_version": "1",
                "model_id": "prostate-classifier",
                "model_version": "1",
                "workflow": "wsi-to-classification-heatmap",
                "resource_profile": "mig20",
                "max_runtime_seconds": 14_400,
                "input_name": "slide",
                "output_name": "heatmap",
                "model_parameter_names": ["threshold"],
                "allowed_wsi_plugin_ids": ["openslide:11880"],
            }
            storage_mapping = {
                "storage_id": "slides",
                "source_prefix": "Projects/empaia",
                "mount_root": temp_dir,
            }
            environment = {
                "EMPAIA_JES_BEARER_TOKEN_FILE": str(token_path),
                "EMPAIA_APP_MAPPINGS_JSON": json.dumps([mapping]),
                "EMPAIA_EXECUTION_DATABASE_PATH": str(database_path),
                "EMPAIA_INSTALLATION_ALIAS": "poc",
                "EMPAIA_APP_SERVICE_URL": "http://app-service:8000",
                "EMPAIA_APP_SERVICE_BEARER_TOKEN_FILE": str(app_service_token_path),
                "EMPAIA_STORAGE_MAPPINGS_JSON": json.dumps([storage_mapping]),
                "EMPAIA_POC_ALLOW_BROAD_APP_SERVICE_CREDENTIAL": "true",
            }

            with patch.dict(os.environ, environment, clear=True):
                settings = Settings.from_env()

        self.assertEqual(token_path, settings.jes_bearer_token_file)
        self.assertEqual(database_path, settings.database_path)
        self.assertEqual("prostate-classifier", settings.app_mappings[0].model_id)
        self.assertEqual(
            ("openslide:11880",),
            settings.app_mappings[0].allowed_wsi_plugin_ids,
        )
        self.assertEqual("slides", settings.storage_mappings[0].storage_id)

    def test_duplicate_app_mapping_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            token_path = Path(temp_dir) / "jes-token"
            token_path.write_text("secret-token", encoding="utf-8")
            app_service_token_path = Path(temp_dir) / "app-service-token"
            app_service_token_path.write_text("app-secret-token", encoding="utf-8")
            mapping = {
                "app_id": "prostate-app",
                "app_version": "1",
                "model_id": "prostate-classifier",
                "model_version": "1",
                "workflow": "wsi-to-classification-heatmap",
                "resource_profile": "mig20",
                "max_runtime_seconds": 14_400,
            }
            environment = {
                "EMPAIA_JES_BEARER_TOKEN_FILE": str(token_path),
                "EMPAIA_APP_MAPPINGS_JSON": json.dumps([mapping, mapping]),
                "EMPAIA_INSTALLATION_ALIAS": "poc",
                "EMPAIA_APP_SERVICE_URL": "http://app-service:8000",
                "EMPAIA_APP_SERVICE_BEARER_TOKEN_FILE": str(app_service_token_path),
                "EMPAIA_STORAGE_MAPPINGS_JSON": json.dumps(
                    [
                        {
                            "storage_id": "slides",
                            "source_prefix": "Projects/empaia",
                            "mount_root": temp_dir,
                        }
                    ]
                ),
            }

            with (
                patch.dict(os.environ, environment, clear=True),
                self.assertRaisesRegex(ValueError, "duplicate app mapping"),
            ):
                Settings.from_env()


if __name__ == "__main__":
    unittest.main()
