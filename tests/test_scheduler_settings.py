"""Check scheduler configuration through environment loading and direct construction."""

from __future__ import annotations

import os
import unittest
from datetime import timedelta
from unittest.mock import patch

from pydantic import ValidationError

from model_service.settings import SchedulerSettings


class SchedulerSettingsTests(unittest.TestCase):
    """Keep environment names, defaults, validation, and source isolation stable."""

    def setUp(self) -> None:
        """Provide synthetic required settings without connecting to any services."""
        self.environment = {
            "DATABASE_URL": "postgresql://unused",
            "PROVIDER_CONTEXT_KEY": "synthetic-key",
            "RAY_SERVE_ORIGIN": "http://workflow-serve",
        }

    def test_defaults_remain_unchanged(self) -> None:
        """Load the established runtime defaults from only the required settings."""
        settings = SchedulerSettings.from_environment(self.environment)
        self.assertEqual(settings.database_url, "postgresql://unused")
        self.assertEqual(settings.ray_serve_origin, "http://workflow-serve")
        self.assertIsNone(settings.inference_serve_origin)
        self.assertEqual(settings.inference_model_origins, {})
        self.assertEqual(settings.inference_timeout, timedelta(minutes=15))
        self.assertEqual(settings.lease_duration, timedelta(seconds=60))
        self.assertEqual(settings.poll_interval, 0.5)
        self.assertEqual(settings.processing_attempts, 3)
        self.assertEqual(settings.provider_finish_attempts, 3)
        self.assertEqual(settings.initial_backoff, timedelta(seconds=5))
        self.assertEqual(settings.maximum_backoff, timedelta(minutes=5))
        self.assertEqual(settings.default_deadline, timedelta(hours=6))
        self.assertEqual(settings.maximum_deadline, timedelta(hours=24))
        self.assertEqual(settings.per_workflow_limit, 1)
        self.assertIsNone(settings.empaia_gateway_factory)

    def test_environment_names_and_numeric_seconds_remain_supported(self) -> None:
        """Parse deployment variable names, fractional seconds, and model-origin JSON."""
        settings = SchedulerSettings.from_environment(
            {
                **self.environment,
                "INFERENCE_SERVE_ORIGIN": " http://model-serve ",
                "INFERENCE_MODEL_ORIGINS": '{"classifier":"http://model-serve"}',
                "INFERENCE_TIMEOUT_SECONDS": "1e3",
                "LEASE_SECONDS": "12.5",
                "WORKER_POLL_SECONDS": "0.25",
                "PROCESSING_ATTEMPTS": "4",
                "PROVIDER_FINISH_ATTEMPTS": "5",
                "INITIAL_BACKOFF_SECONDS": "0",
                "MAXIMUM_BACKOFF_SECONDS": "30",
                "DEFAULT_DEADLINE_SECONDS": "120",
                "MAXIMUM_DEADLINE_SECONDS": "240",
                "PER_WORKFLOW_LIMIT": "2",
                "EMPAIA_GATEWAY_FACTORY": "integrations.empaia.http_gateway:create_http_gateway",
                "UNRELATED_SETTING": "ignored",
            }
        )
        self.assertEqual(settings.inference_serve_origin, "http://model-serve")
        self.assertEqual(
            str(settings.inference_model_origins["classifier"]), "http://model-serve/"
        )
        self.assertEqual(settings.inference_timeout, timedelta(seconds=1000))
        self.assertEqual(settings.lease_duration, timedelta(seconds=12.5))
        self.assertEqual(settings.poll_interval, 0.25)
        self.assertEqual(settings.processing_attempts, 4)
        self.assertEqual(settings.provider_finish_attempts, 5)
        self.assertEqual(settings.initial_backoff, timedelta(0))
        self.assertEqual(settings.maximum_backoff, timedelta(seconds=30))
        self.assertEqual(settings.default_deadline, timedelta(seconds=120))
        self.assertEqual(settings.maximum_deadline, timedelta(seconds=240))
        self.assertEqual(settings.per_workflow_limit, 2)
        self.assertEqual(
            settings.empaia_gateway_factory,
            "integrations.empaia.http_gateway:create_http_gateway",
        )

    def test_process_environment_is_used_only_when_requested(self) -> None:
        """Read process variables when no explicit mapping is supplied."""
        with patch.dict(os.environ, self.environment, clear=True):
            self.assertEqual(
                SchedulerSettings.from_environment().database_url, "postgresql://unused"
            )

    def test_explicit_mapping_never_falls_back_to_process_environment(self) -> None:
        """Do not fill missing explicit settings, even for an empty mapping."""
        with patch.dict(
            os.environ, {**self.environment, "WORKER_POLL_SECONDS": "9"}, clear=True
        ):
            for values in ({}, {"DATABASE_URL": "postgresql://other"}):
                with self.subTest(values=values), self.assertRaises(ValidationError):
                    SchedulerSettings.from_environment(values)
            self.assertEqual(
                SchedulerSettings.from_environment(self.environment).poll_interval, 0.5
            )
            self.assertEqual(self._direct_settings().poll_interval, 0.5)

    def test_required_settings_reject_missing_or_blank_values(self) -> None:
        """Validate required values before runtime wiring can use them."""
        for name in self.environment:
            missing = dict(self.environment)
            del missing[name]
            for values in (missing, {**self.environment, name: " \t "}):
                with self.subTest(name=name), self.assertRaises(ValidationError):
                    SchedulerSettings.from_environment(values)

    def test_blank_optional_settings_are_absent(self) -> None:
        """Treat whitespace-only optional settings as unset."""
        settings = SchedulerSettings.from_environment(
            {
                **self.environment,
                "INFERENCE_SERVE_ORIGIN": " \t ",
                "EMPAIA_GATEWAY_FACTORY": "",
            }
        )
        self.assertIsNone(settings.inference_serve_origin)
        self.assertIsNone(settings.empaia_gateway_factory)

    def test_numeric_environment_fields_reject_invalid_bounds_and_nonfinite_values(
        self,
    ) -> None:
        """Reject invalid counts and durations, including NaN, infinity, and overflow."""
        positive = (
            "INFERENCE_TIMEOUT_SECONDS",
            "LEASE_SECONDS",
            "WORKER_POLL_SECONDS",
            "DEFAULT_DEADLINE_SECONDS",
            "MAXIMUM_DEADLINE_SECONDS",
        )
        for name in (*positive, "INITIAL_BACKOFF_SECONDS", "MAXIMUM_BACKOFF_SECONDS"):
            invalid_values = ["-1", "nan", "inf", "-inf", "invalid"]
            if name != "WORKER_POLL_SECONDS":
                invalid_values.append("1e300")
            for value in invalid_values:
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaises(ValueError),
                ):
                    SchedulerSettings.from_environment(
                        {**self.environment, name: value}
                    )
        for name in (
            *positive,
            "PROCESSING_ATTEMPTS",
            "PROVIDER_FINISH_ATTEMPTS",
            "PER_WORKFLOW_LIMIT",
        ):
            with self.subTest(name=name), self.assertRaises(ValidationError):
                SchedulerSettings.from_environment({**self.environment, name: "0"})
        for name in (
            "PROCESSING_ATTEMPTS",
            "PROVIDER_FINISH_ATTEMPTS",
            "PER_WORKFLOW_LIMIT",
        ):
            for value in ("-1", "1.5", "invalid"):
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaises(ValidationError),
                ):
                    SchedulerSettings.from_environment(
                        {**self.environment, name: value}
                    )

    def test_policy_bounds_apply_to_environment_and_direct_construction(self) -> None:
        """Apply ordering constraints regardless of the configuration source."""
        for overrides in (
            {"INITIAL_BACKOFF_SECONDS": "301"},
            {"DEFAULT_DEADLINE_SECONDS": "86401"},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValidationError):
                SchedulerSettings.from_environment({**self.environment, **overrides})
        for overrides in (
            {"initial_backoff": timedelta(seconds=301)},
            {"default_deadline": timedelta(hours=25)},
            {"lease_duration": timedelta(0)},
            {"processing_attempts": 0},
            {"poll_interval": float("nan")},
            {"poll_interval": float("inf")},
            {"inference_model_origins": {"invalid/path": "http://model-serve"}},
        ):
            with self.subTest(overrides=overrides), self.assertRaises(ValidationError):
                self._direct_settings(**overrides)

    def test_direct_construction_retains_typed_durations_and_frozen_fields(
        self,
    ) -> None:
        """Keep typed constructor values and reject changes after validation."""
        settings = self._direct_settings(lease_duration=timedelta(seconds=12.5))
        self.assertEqual(settings.lease_duration, timedelta(seconds=12.5))
        with self.assertRaises(ValidationError):
            settings.poll_interval = 2

    def _direct_settings(self, **overrides: object) -> SchedulerSettings:
        """Construct settings by Python field names, independent of environment loading."""
        return SchedulerSettings(
            **{
                "database_url": "postgresql://unused",
                "provider_context_key": "synthetic-key",
                "ray_serve_origin": "http://workflow-serve",
                **overrides,
            }
        )


if __name__ == "__main__":
    unittest.main()
