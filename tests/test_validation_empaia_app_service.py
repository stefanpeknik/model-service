"""Check mock AppService scenario behavior and recorded publication state."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from validation.empaia_app_service import ValidationScenario, app


class ValidationAppServiceTests(unittest.TestCase):
    """Exercise mock AppService scenario behavior and recorded publication state."""

    def test_configuration_scenarios_are_explicit(self) -> None:
        """Verify configuration scenarios are explicit.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        headers = {"Authorization": "Bearer synthetic-token"}
        with TestClient(app) as client:
            invalid = client.get(
                f"/v3/{ValidationScenario.INVALID_CONFIGURATION}-contract/configuration",
                headers=headers,
            )
            self.assertEqual(invalid.status_code, 200)
            self.assertEqual(invalid.json()["customer"], {"unexpected": True})

            unavailable = client.get(
                f"/v3/{ValidationScenario.SERVICE_UNAVAILABLE}-contract/configuration",
                headers=headers,
            )
            self.assertEqual(unavailable.status_code, 503)

    def test_workflow_failure_points_at_an_unavailable_slide(self) -> None:
        """Verify workflow failure points at an unavailable slide.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        job_id = f"{ValidationScenario.WORKFLOW_FAILURE}-contract"
        headers = {"Authorization": "Bearer synthetic-token"}
        with TestClient(app) as client:
            direct = client.get(
                f"/v3/rationai/{job_id}/inputs/slide/direct-access", headers=headers
            )
            self.assertEqual(direct.status_code, 200)
            self.assertTrue(direct.json()["storage_path"].endswith("missing-slide.dat"))
            failed = client.put(
                f"/v3/{job_id}/failure",
                headers=headers,
                json={"user_message": "safe failure"},
            )
            self.assertEqual(failed.status_code, 204)
            recorded = client.get(f"/validation/jobs/{job_id}")
            self.assertEqual(recorded.json()["direct_access_reads"], 1)
            self.assertEqual(recorded.json()["failure_reports"], 1)
            self.assertEqual(recorded.json()["failure"], "safe failure")

    def test_publication_and_finalization_failures_are_counted(self) -> None:
        """Verify publication and finalization failures are counted.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        headers = {"Authorization": "Bearer synthetic-token"}
        with TestClient(app) as client:
            publication_job = f"{ValidationScenario.PUBLICATION_FAILURE}-contract"
            created = client.post(
                f"/v3/{publication_job}/outputs/heatmap",
                headers=headers,
                json={"type": "continuous_pixelmap"},
            )
            tile = client.put(
                f"/v3/{publication_job}/pixelmaps/{created.json()['id']}"
                "/level/0/position/0/0/data",
                headers=headers,
                content=b"tile",
            )
            self.assertEqual(tile.status_code, 503)
            publication = client.get(f"/validation/jobs/{publication_job}").json()
            self.assertEqual(publication["tile_attempts"], 1)
            self.assertEqual(publication["tile_count"], 0)

            finalization_job = f"{ValidationScenario.FINALIZATION_FAILURE}-contract"
            created = client.post(
                f"/v3/{finalization_job}/outputs/heatmap",
                headers=headers,
                json={"type": "continuous_pixelmap"},
            )
            tile = client.put(
                f"/v3/{finalization_job}/pixelmaps/{created.json()['id']}"
                "/level/0/position/0/0/data",
                headers=headers,
                content=b"tile",
            )
            self.assertEqual(tile.status_code, 204)
            finalized = client.put(f"/v3/{finalization_job}/finalize", headers=headers)
            self.assertEqual(finalized.status_code, 503)
            finalization = client.get(f"/validation/jobs/{finalization_job}").json()
            self.assertEqual(finalization["tile_count"], 1)
            self.assertEqual(finalization["finalize_attempts"], 1)

    def test_scheduler_restart_delays_only_the_first_slide_access(self) -> None:
        """Verify scheduler restart delays only the first slide access.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        job_id = f"{ValidationScenario.SCHEDULER_RESTART}-contract"
        headers = {"Authorization": "Bearer synthetic-token"}
        path = f"/v3/rationai/{job_id}/inputs/slide/direct-access"
        with (
            patch(
                "validation.empaia_app_service.asyncio.sleep",
                new_callable=AsyncMock,
            ) as sleep,
            TestClient(app) as client,
        ):
            first = client.get(path, headers=headers)
            second = client.get(path, headers=headers)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        sleep.assert_awaited_once_with(45)

    def test_ray_restart_delays_only_the_first_slide_access(self) -> None:
        """Verify Ray restart delays only the first slide access.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        job_id = f"{ValidationScenario.RAY_RESTART}-contract"
        headers = {"Authorization": "Bearer synthetic-token"}
        path = f"/v3/rationai/{job_id}/inputs/slide/direct-access"
        with (
            patch(
                "validation.empaia_app_service.asyncio.sleep",
                new_callable=AsyncMock,
            ) as sleep,
            TestClient(app) as client,
        ):
            first = client.get(path, headers=headers)
            second = client.get(path, headers=headers)

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        sleep.assert_awaited_once_with(45)

    def test_rectangle_input_has_no_direct_access(self) -> None:
        """Verify rectangle input has no direct access.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        job_id = f"{ValidationScenario.RECTANGLE}-contract"
        headers = {"Authorization": "Bearer synthetic-token"}
        with TestClient(app) as client:
            direct = client.get(
                f"/v3/rationai/{job_id}/inputs/region_of_interest/direct-access",
                headers=headers,
            )
            self.assertEqual(direct.status_code, 404)
            rectangle = client.get(
                f"/v3/{job_id}/inputs/region_of_interest", headers=headers
            )
            self.assertEqual(rectangle.status_code, 200)
            self.assertEqual(
                rectangle.json(),
                {"upper_left": [1, 1], "width": 6, "height": 6},
            )
            recorded = client.get(f"/validation/jobs/{job_id}")
            self.assertEqual(recorded.json()["rectangle_reads"], 1)

    def test_records_public_contract(self) -> None:
        """Verify records public contract.

        Returns:
            None: No return value.

        Raises:
            AssertionError: An expected invariant or fixture interaction does not hold.
        """
        job_id = "validation-contract"
        headers = {"Authorization": "Bearer synthetic-token"}
        with TestClient(app) as client:
            configuration = client.get(f"/v3/{job_id}/configuration", headers=headers)
            self.assertEqual(configuration.status_code, 200)
            self.assertEqual(configuration.json(), {"global": {}, "customer": {}})

            direct = client.get(
                f"/v3/rationai/{job_id}/inputs/slide/direct-access", headers=headers
            )
            self.assertEqual(direct.status_code, 200)
            self.assertEqual(
                direct.json()["storage_path"], "/mnt/validation/input/slide.dat"
            )

            created = client.post(
                f"/v3/{job_id}/outputs/heatmap",
                headers=headers,
                json={"type": "continuous_pixelmap"},
            )
            self.assertEqual(created.status_code, 200)
            pixelmap_id = created.json()["id"]
            tile = client.put(
                f"/v3/{job_id}/pixelmaps/{pixelmap_id}/level/0/position/0/0/data",
                headers=headers,
                content=b"tile-data",
            )
            self.assertEqual(tile.status_code, 204)
            finalized = client.put(f"/v3/{job_id}/finalize", headers=headers)
            self.assertEqual(finalized.status_code, 204)

            recorded = client.get(f"/validation/jobs/{job_id}")
            self.assertEqual(recorded.status_code, 200)
            self.assertEqual(recorded.json()["configuration_reads"], 1)
            self.assertEqual(recorded.json()["tile_count"], 1)
            self.assertTrue(recorded.json()["finalized"])
