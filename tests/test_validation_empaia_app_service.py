from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from validation.empaia_app_service import app


class ValidationAppServiceTests(unittest.TestCase):
    def test_rectangle_input_has_no_direct_access(self) -> None:
        job_id = "rectangle-contract"
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
