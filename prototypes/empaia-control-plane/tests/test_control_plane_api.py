"""Check historical prototype authentication and safe public lifecycle responses.

These cases describe the archived SQLite prototype rather than the current production
scheduler.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient
from httpx import Response

from control_plane.api import create_app
from control_plane.domain import AppMapping
from control_plane.settings import Settings
from control_plane.storage import StorageMapping


class ControlPlaneApiTestCase(unittest.TestCase):
    """Exercise historical prototype authentication and safe public lifecycle responses."""

    def setUp(self) -> None:
        """Create isolated fixtures and clients for historical prototype authentication and safe public lifecycle responses.

        Returns:
            None: No return value.
        """
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        root = Path(self.temp_dir.name)
        jes_token = root / "jes-token"
        app_service_token = root / "app-service-token"
        jes_token.write_text("test-service-token", encoding="utf-8")
        app_service_token.write_text("test-app-service-token", encoding="utf-8")
        settings = Settings(
            database_path=root / "jobs.db",
            installation_alias="poc",
            jes_bearer_token_file=jes_token,
            app_mappings=(
                AppMapping(
                    app_id="prostate-app",
                    app_version="1",
                    model_id="prostate-classifier",
                    model_version="1",
                    workflow="wsi-to-classification-heatmap",
                    resource_profile="mig20",
                    max_runtime_seconds=14_400,
                    allowed_wsi_plugin_ids=("openslide:11880",),
                ),
            ),
            app_service_url="https://app-service.example",
            app_service_token_file=app_service_token,
            storage_mappings=(StorageMapping("slides", "Projects/empaia", root),),
            allow_broad_app_service_credential=True,
        )
        self.client = TestClient(create_app(settings))
        self.jes_token_file = jes_token
        self.headers = {"Authorization": "Bearer test-service-token"}
        self.submission = {
            "empaia_job_id": "job-123",
            "registered_app_id": "prostate-app",
            "registered_app_version": "1",
            "installation_alias": "poc",
            "idempotency_key": "poc:job-123:prostate-app:1",
        }

    def request(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, str] | None = None,
        authenticated: bool = False,
    ) -> Response:
        """Send a prototype ASGI request with optional authentication and JSON body.

        Args:
            method: HTTP method used by the request or test client.
            path: File, storage locator, or HTTP endpoint selected by this operation.
            json_body: Optional prototype request payload sent through its ASGI test
                client.
            authenticated: Whether to attach the configured JES test bearer header.

        Returns:
            Response: HTTP response returned by the transport or controlled scenario
                handler.
        """
        return self.client.request(
            method,
            path,
            json=json_body,
            headers=self.headers if authenticated else None,
        )

    def test_provider_routes_require_authentication(self) -> None:
        """Verify provider routes require authentication.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        response = self.request(
            "POST", "/v1/empaia/executions", json_body=self.submission
        )

        self.assertEqual(401, response.status_code)
        self.assertEqual("Bearer", response.headers["www-authenticate"])

        self.jes_token_file.write_text("rotated-service-token", encoding="utf-8")
        old_token = self.request(
            "POST",
            "/v1/empaia/executions",
            json_body=self.submission,
            authenticated=True,
        )
        rotated_token = self.client.request(
            "POST",
            "/v1/empaia/executions",
            json=self.submission,
            headers={"Authorization": "Bearer rotated-service-token"},
        )
        self.assertEqual(401, old_token.status_code)
        self.assertEqual(202, rotated_token.status_code)

    def test_submission_is_idempotent_and_status_is_safe(self) -> None:
        """Verify submission is idempotent and status is safe.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        first = self.request(
            "POST",
            "/v1/empaia/executions",
            json_body=self.submission,
            authenticated=True,
        )
        duplicate = self.request(
            "POST",
            "/v1/empaia/executions",
            json_body=self.submission,
            authenticated=True,
        )

        self.assertEqual(202, first.status_code)
        self.assertEqual(first.json()["execution_id"], duplicate.json()["execution_id"])
        self.assertEqual(first.json()["status_url"], first.headers["location"])

        status_response = self.request(
            "GET", first.json()["status_url"], authenticated=True
        )
        self.assertEqual(200, status_response.status_code)
        self.assertEqual(
            {
                "execution_id",
                "empaia_job_id",
                "state",
                "cancellation_requested",
                "created_at",
                "updated_at",
                "ready_at",
                "first_claimed_at",
                "failure_category",
                "timeout_kind",
                "reason",
            },
            set(status_response.json()),
        )

    def test_cancellation_is_idempotent(self) -> None:
        """Verify cancellation is idempotent.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        submitted = self.request(
            "POST",
            "/v1/empaia/executions",
            json_body=self.submission,
            authenticated=True,
        )
        cancellation_url = f"{submitted.json()['status_url']}/cancellation"

        first = self.request("POST", cancellation_url, authenticated=True)
        second = self.request("POST", cancellation_url, authenticated=True)

        self.assertEqual(202, first.status_code)
        self.assertEqual("CANCELLED", first.json()["state"])
        self.assertEqual(first.json(), second.json())


if __name__ == "__main__":
    unittest.main()
