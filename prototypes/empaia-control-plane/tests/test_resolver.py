"""Check historical prototype resolution into an authorized credential-free worker specification.

These cases describe the archived SQLite prototype rather than the current production
scheduler.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from control_plane.app_service import DirectAccess, JobConfiguration
from control_plane.domain import AdmissionRequest, AppMapping, ExecutionState
from control_plane.registry import AppRegistry
from control_plane.resolver import EmpaiaExecutionResolver
from control_plane.service import ExecutionService
from control_plane.storage import (
    InvalidStorageInputError,
    StorageLocatorResolver,
    StorageMapping,
)
from control_plane.store import SQLiteExecutionStore


class StubAppService:
    """Return fixed configuration and direct-access metadata for prototype resolver tests."""

    def __init__(self, storage_path: str, plugin_id: str = "openslide:11880") -> None:
        """Configure provider storage/plugin metadata and initialize recorded failure messages.

        Args:
            storage_path: Provider storage locator to authorize or expose in fixture
                metadata.
            plugin_id: WSI reader plugin identity exposed by prototype fixture metadata.
        """
        self.storage_path = storage_path
        self.plugin_id = plugin_id
        self.reported_failures: list[str] = []

    def get_configuration(self, _job_id: str) -> JobConfiguration:
        """Return the required threshold plus an unrelated value for allow-list selection tests.

        Args:
            _job_id: Unused external empaia job identifier; retained to match the
                interface.

        Returns:
            JobConfiguration: Validated global and customer provider configuration
                sections.
        """
        return JobConfiguration({"threshold": 0.5, "ignored": True}, {})

    def get_direct_access(self, _job_id: str, _input_name: str) -> DirectAccess:
        """Return fixed slide identity and configured shared storage/plugin information.

        Args:
            _job_id: Unused external empaia job identifier; retained to match the
                interface.
            _input_name: Unused named external job input to resolve; retained to match
                the interface.

        Returns:
            DirectAccess: Validated provider shared path and direct-access support
                metadata.
        """
        return DirectAccess(
            slide_id="wsi-123",
            required_plugin_id=self.plugin_id,
            storage_path=self.storage_path,
            supported=True,
        )

    def report_failure(self, _job_id: str, user_message: str) -> None:
        """Record the safe message sent by prototype resolution.

        Args:
            _job_id: Unused external empaia job identifier; retained to match the
                interface.
            user_message: Safe external job failure message, bounded by the provider
                contract.

        Returns:
            None: No return value.
        """
        self.reported_failures.append(user_message)


class ResolverTestCase(unittest.TestCase):
    """Exercise historical prototype resolution into an authorized credential-free worker specification."""

    def test_resolution_builds_allowlisted_credential_free_spec(self) -> None:
        """Verify resolution builds allowlisted credential free spec.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            slide = root / "job_slides" / "slide.tiff"
            slide.parent.mkdir()
            slide.write_bytes(b"wsi")
            store = SQLiteExecutionStore(root / "jobs.db")
            mapping = AppMapping(
                app_id="prostate-app",
                app_version="1",
                model_id="prostate-classifier",
                model_version="1",
                workflow="wsi-to-classification-heatmap",
                resource_profile="mig20",
                max_runtime_seconds=14_400,
                model_parameter_names=("threshold",),
                allowed_wsi_plugin_ids=("openslide:11880",),
            )
            service = ExecutionService(store, AppRegistry([mapping]), "poc")
            storage = StorageLocatorResolver(
                (
                    StorageMapping(
                        storage_id="slides",
                        source_prefix="Projects/empaia",
                        mount_root=root,
                    ),
                )
            )
            execution = service.admit(
                AdmissionRequest(
                    installation_alias="poc",
                    empaia_job_id="job-123",
                    registered_app_id="prostate-app",
                    registered_app_version="1",
                    idempotency_key="poc:job-123:prostate-app:1",
                )
            ).execution
            resolver = EmpaiaExecutionResolver(
                store,
                StubAppService("Projects/empaia/job_slides/slide.tiff"),
                storage,
                retry_seconds=5,
            )

            result = resolver.run_once()
            resolved = store.get(execution.id)

            unsupported = service.admit(
                AdmissionRequest(
                    installation_alias="poc",
                    empaia_job_id="job-unsupported",
                    registered_app_id="prostate-app",
                    registered_app_version="1",
                    idempotency_key="poc:job-unsupported:prostate-app:1",
                )
            ).execution
            unsupported_app = StubAppService(
                "Projects/empaia/job_slides/slide.tiff", plugin_id="other:1"
            )
            EmpaiaExecutionResolver(
                store,
                unsupported_app,
                storage,
                retry_seconds=5,
            ).run_once()
            rejected = store.get(unsupported.id)
            (slide.parent / "linked.tiff").symlink_to(slide)
            with self.assertRaises(InvalidStorageInputError):
                storage.resolve("Projects/empaia/job_slides/linked.tiff")

        self.assertEqual(1, result.ready)
        self.assertEqual(ExecutionState.READY, resolved.state)
        self.assertIsNotNone(resolved.execution_spec)
        assert resolved.execution_spec is not None
        self.assertEqual(
            "job_slides/slide.tiff", resolved.execution_spec.relative_locator
        )
        self.assertEqual({"threshold": 0.5}, resolved.execution_spec.model_parameters)
        self.assertEqual(ExecutionState.FAILED, rejected.state)
        self.assertEqual(1, len(unsupported_app.reported_failures))


if __name__ == "__main__":
    unittest.main()
