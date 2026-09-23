from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from validation.empaia_app_service import ValidationScenario, app


class ValidationAppServiceTests(unittest.TestCase):
    def test_configuration_scenarios_are_explicit(self) -> None:
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

    def test_scheduler_restart_delays_only_the_first_slide_access(self) -> None:
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
